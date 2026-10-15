;;; scenario_org.el --- Org cells for a scenario run -*- lexical-binding: t; -*-
(require 'org)
(require 'org-element)
(require 'ob-core)
(require 'cl-lib)
(require 'subr-x)

(defvar-local torch-dojo-scenario-org-signature nil)
(defvar-local torch-dojo-scenario-org-skipped 0)

(defun torch-dojo-scenario-org-heading (block)
  "Return BLOCK's heading path from its parsed Org ancestors."
  (let ((node (org-element-property :parent block))
        (titles nil))
    (while node
      (when (eq (org-element-type node) 'headline)
        (push (org-element-property :raw-value node) titles))
      (setq node (org-element-property :parent node)))
    (string-join titles " / ")))

(defun torch-dojo-scenario-org-context (block)
  "Return the closest prose paragraph before BLOCK in its section."
  (let ((section (org-element-lineage block '(section)))
        (begin (org-element-property :begin block))
        (paragraph nil))
    (when section
      (org-element-map section 'paragraph
        (lambda (item)
          (when (< (org-element-property :begin item) begin)
            (setq paragraph item)))))
    (if paragraph
        (string-trim (buffer-substring-no-properties
                      (org-element-property :begin paragraph)
                      (org-element-property :end paragraph)))
      "")))

(defun torch-dojo-scenario-org-parse ()
  "Return discovered Python cells and a source-block structure signature."
  (let ((cells nil) (signature nil) (skipped 0))
    (org-element-map (org-element-parse-buffer) 'src-block
      (lambda (block)
        (let* ((language (downcase (or (org-element-property :language block) "")))
               (params (or (org-element-property :parameters block) ""))
               (begin (org-element-property :begin block))
               (line (line-number-at-pos begin))
               (heading (torch-dojo-scenario-org-heading block))
               (name (org-element-property :name block)))
          (let* ((bounds (save-excursion
                           (goto-char begin)
                           (unless (re-search-forward "^[[:space:]]*#\\+BEGIN_SRC[[:space:]]+" (org-element-property :end block) t)
                             (error "Missing source block at line %d" line))
                           (forward-line 1)
                           (let ((start (point)))
                             (unless (re-search-forward "^[[:space:]]*#\\+END_SRC[[:space:]]*$" nil t)
                               (error "Unterminated source block at line %d" line))
                             (cons start (match-beginning 0)))))
                 (source-start (car bounds))
                 (source-end (cdr bounds)))
          (push (list language params name heading) signature)
          (if (not (equal language "python"))
              (cl-incf skipped)
            (when (string-match-p "\\(?:^\\|[[:space:]]\\):\\(?:var\\|session\\|noweb\\|dir\\|eval\\)\\(?:[[:space:]=]\\|$\\)" params)
              (error "Unsupported Python header at %s line %d" heading line))
            (let ((id (format "cell-%d" (1+ (length cells)))))
              (push `((id . ,id) (name . ,(or name ""))
                      (heading . ,heading) (text . ,(torch-dojo-scenario-org-context block))
                      (source . ,(buffer-substring-no-properties source-start source-end))
                      (line . ,line) (begin . ,begin)
                      (contents-begin . ,source-start)
                      (contents-end . ,source-end)
                      (end . ,(org-element-property :end block))) cells)))))))
    (list (vconcat (nreverse cells)) (nreverse signature) skipped)))

(defun torch-dojo-scenario-org-discover ()
  "Discover source cells and remember their block structure."
  (pcase-let ((`(,cells ,signature ,skipped) (torch-dojo-scenario-org-parse)))
    (when (= (length cells) 0) (error "Org file has no Python source blocks"))
    (setq torch-dojo-scenario-org-signature signature
          torch-dojo-scenario-org-skipped skipped)
    cells))

(defun torch-dojo-scenario-org-skipped-count () torch-dojo-scenario-org-skipped)

(defun torch-dojo-scenario-org-current-cells ()
  (pcase-let ((`(,cells ,signature ,_) (torch-dojo-scenario-org-parse)))
    (unless (equal signature torch-dojo-scenario-org-signature)
      (error "Org block structure changed; restart with a new run"))
    cells))

(defun torch-dojo-scenario-org-cell (id)
  (let* ((cells (torch-dojo-scenario-org-current-cells))
         (cell (cl-find id cells :key (lambda (item) (alist-get 'id item)) :test #'equal)))
    (unless cell (error "Unknown cell ID: %s" id))
    cell))

(defun torch-dojo-scenario-org-source (id)
  (alist-get 'source (torch-dojo-scenario-org-cell id)))

(defun torch-dojo-scenario-org-set-source (id source)
  (unless (and (stringp source) (not (string-match-p "\r" source))
               (not (string-match-p "^[[:space:]]*#\\+\\(?:BEGIN_SRC\\|END_SRC\\|NAME:\\)" source)))
    (error "Invalid source block boundary"))
  (let* ((cell (torch-dojo-scenario-org-cell id))
         (start (alist-get 'contents-begin cell))
         (end (alist-get 'contents-end cell)))
    (save-excursion
      (delete-region start end)
      (goto-char start)
      (insert (if (or (string-empty-p source) (string-suffix-p "\n" source))
                  source (concat source "\n")))))
  (when (buffer-file-name) (save-buffer)))

(defun torch-dojo-scenario-org-result-position (cell)
  "Move point after CELL's end directive and any separating blank lines."
  (goto-char (alist-get 'contents-end cell))
  (unless (re-search-forward "^[[:space:]]*#\\+END_SRC[[:space:]]*$" nil t)
    (error "Missing block end for %s" (alist-get 'id cell)))
  (goto-char (match-beginning 0))
  (forward-line 1)
  (while (and (not (eobp)) (looking-at "^[[:space:]]*$"))
    (forward-line 1)))

(defun torch-dojo-scenario-org-remove-result (cell)
  "Remove the Org result immediately following CELL, if present."
  (torch-dojo-scenario-org-result-position cell)
  (when (looking-at "^[[:space:]]*#\\+RESULTS\\(?:\\[[^]]*\\]\\)?:.*$")
    (let ((start (point)))
      (forward-line 1)
      (delete-region start (org-babel-result-end)))))

(defun torch-dojo-scenario-org-write-result (id result)
  "Write a plain-text RESULT under ID's source block."
  (let ((cell (torch-dojo-scenario-org-cell id)))
    (save-excursion
      (torch-dojo-scenario-org-remove-result cell)
      (unless (bolp) (insert "\n"))
      (insert (format "#+RESULTS:\n: %s\n"
                      (replace-regexp-in-string "\n" "\n: " (string-trim-right result))))))
  (when (buffer-file-name) (save-buffer)))

(defun torch-dojo-scenario-org-clear-results ()
  (let ((cells (torch-dojo-scenario-org-current-cells)))
    (cl-loop for i downfrom (1- (length cells)) to 0
             do (save-excursion
                  (torch-dojo-scenario-org-remove-result (aref cells i)))))
  (when (buffer-file-name) (save-buffer)))

(provide 'scenario_org)
