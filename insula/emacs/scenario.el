;;; scenario.el --- Stepwise Jupyter scenario in Org -*- lexical-binding: t; -*-
(require 'cl-lib)
(require 'org)
(require 'subr-x)
(require 'json)
(require 'url)
(load-file "/workspace/torch_dojo/insula/emacs/scenario_org.el")

(defvar-local torch-dojo-scenario-steps nil)
(defvar-local torch-dojo-scenario-status-text "Ready")
(defvar-local torch-dojo-scenario-output nil)
(defvar-local torch-dojo-scenario-submitted nil)
(defvar-local torch-dojo-scenario-stale nil)

(defun torch-dojo-scenario-step (index)
  (aref torch-dojo-scenario-steps index))

(defun torch-dojo-scenario-step-name (index)
  (alist-get 'name (torch-dojo-scenario-step index)))

(defun torch-dojo-scenario-step-kind (index)
  (alist-get 'kind (torch-dojo-scenario-step index)))

(defvar-local torch-dojo-scenario-index 0)
(defvar-local torch-dojo-scenario-client nil)
(defvar-local torch-dojo-scenario-kernel-id nil)
(defvar-local torch-dojo-scenario-busy nil)
(defvar-local torch-dojo-scenario-failed nil)
(defvar-local torch-dojo-scenario-last-error nil)
(defvar-local torch-dojo-scenario-history nil)
(defvar-local torch-dojo-scenario-view-index nil)
(defvar-local torch-dojo-scenario-change-hook nil)
(defvar-local torch-dojo-scenario-before-exit-hook nil)

(defun torch-dojo-scenario-snapshot ()
  "Return the currently visible state without executing anything."
  (let* ((cells (torch-dojo-scenario-org-current-cells))
         (index torch-dojo-scenario-index)
         (view (or torch-dojo-scenario-view-index index))
         (steps
          (cl-loop for i below (length torch-dojo-scenario-steps)
                   for step = (torch-dojo-scenario-step i)
                   for id = (alist-get 'cell_id step)
                   for cell = (and id (cl-find id cells :key (lambda (item) (alist-get 'id item)) :test #'equal))
                   for output = (cdr (assq i torch-dojo-scenario-output))
                   do (when (and cell (assoc id torch-dojo-scenario-submitted)
                                 (not (equal (alist-get 'source cell)
                                             (cdr (assoc id torch-dojo-scenario-submitted)))))
                        (cl-pushnew id torch-dojo-scenario-stale :test #'equal))
                   collect `((stale . ,(if (member id torch-dojo-scenario-stale) t :false))
                             (kind . ,(symbol-name (alist-get 'kind step)))
                             (cell_id . ,(or id :null))
                             (name . ,(alist-get 'name step))
                             (block_name . ,(if (and cell (not (string-empty-p (alist-get 'name cell))))
                                                (alist-get 'name cell) :null))
                             (heading . ,(if cell (alist-get 'heading cell) :null))
                             (text . ,(if cell (alist-get 'text cell) ""))
                             (line . ,(if cell (alist-get 'line cell) :null))
                             (state . ,(cond ((< i index) "complete")
                                             ((and (= i index) torch-dojo-scenario-failed) "failed")
                                             ((= i index) "pending")
                                             (t "locked")))
                             (source . ,(if cell (alist-get 'source cell) :null))
                             (output . ,(if output
                                            `((stdout . ,(or (plist-get output :stdout) ""))
                                              (text . ,(or (plist-get output :text) ""))
                                              (mime . ,(vconcat (or (plist-get output :mime) nil)))
                                              (error . ,(or (plist-get output :error) :null)))
                                          :null))
                             (result . ,(or (cdr (assq i torch-dojo-scenario-history))
                                            (and (= i index) torch-dojo-scenario-last-error)
                                            :null))))))
    `((steps . ,(vconcat steps))
      (skipped_count . ,(torch-dojo-scenario-org-skipped-count))
      (pending_index . ,index)
      (viewed_index . ,(if (< view (length torch-dojo-scenario-steps)) view :null))
      (busy . ,(if torch-dojo-scenario-busy t :false))
      (failed . ,(if torch-dojo-scenario-failed t :false))
      (emacs_pid . ,(emacs-pid))
      (status . ,torch-dojo-scenario-status-text)
      (kernel_id . ,(or torch-dojo-scenario-kernel-id :null))
      (run_file . ,(buffer-file-name)))))

(defun torch-dojo-scenario-set-source (id source)
  "Save ID's source while retaining the freshness of its recorded result."
  (let ((before (torch-dojo-scenario-org-source id)))
    (torch-dojo-scenario-org-set-source id source)
    (when (and (not (equal before (torch-dojo-scenario-org-source id)))
               (cl-loop for i below (length torch-dojo-scenario-steps)
                        thereis (and (equal id (alist-get 'cell_id (torch-dojo-scenario-step i)))
                                     (or (< i torch-dojo-scenario-index)
                                         (assq i torch-dojo-scenario-output)))))
      (cl-pushnew id torch-dojo-scenario-stale :test #'equal))))

(defun torch-dojo-scenario-view-completed (index)
  "Browse completed INDEX without changing execution progress."
  (unless (and (integerp index) (<= 0 index) (< index torch-dojo-scenario-index))
    (user-error "Choose a completed checkpoint to review"))
  (setq torch-dojo-scenario-view-index index)
  (torch-dojo-scenario-focus index))

(defvar torch-dojo-scenario-mode-map
  (let ((map (make-sparse-keymap)))
    (set-keymap-parent map org-mode-map)
    (define-key map (kbd "n") #'torch-dojo-scenario-next)
    (define-key map (kbd "p") #'torch-dojo-scenario-previous)
    (define-key map (kbd "r") #'torch-dojo-scenario-restart)
    (define-key map (kbd "q") #'torch-dojo-scenario-quit)
    map))

(define-derived-mode torch-dojo-scenario-mode org-mode "Dojo-Scenario"
  "Major mode for the stepwise Torch Dojo scenario.")

(defun torch-dojo-scenario-dir ()
  "Return the scenario run directory inside the mutable repo."
  (let ((dir (getenv "DOJO_SCENARIO_DIR")))
    (unless (and dir (file-directory-p dir)
                 (string-prefix-p "/workspace/torch_dojo/target/insula/scenario/" dir))
      (error "DOJO_SCENARIO_DIR is missing or outside scenario state"))
    dir))

(defun torch-dojo-scenario-open ()
  "Open a run-specific Org file and enable scenario controls."
  (interactive)
  (find-file (expand-file-name "run.org" (torch-dojo-scenario-dir)))
  (torch-dojo-scenario-mode)
  (let ((cells (torch-dojo-scenario-org-discover)))
    (setq-local torch-dojo-scenario-steps
                (vconcat (list '((kind . setup) (name . "Setup"))
                               '((kind . connect) (name . "Connect")))
                         (mapcar (lambda (cell)
                                   `((kind . cell) (name . ,(string-join
                                                            (delq nil (list
                                                                       (unless (string-empty-p (alist-get 'heading cell))
                                                                         (alist-get 'heading cell))
                                                                       (if (string-empty-p (alist-get 'name cell))
                                                                           (alist-get 'id cell)
                                                                         (alist-get 'name cell))))
                                                            " · "))
                                     (cell_id . ,(alist-get 'id cell))))
                                 (append cells nil))
                         (list '((kind . final) (name . "Final"))
                               '((kind . cleanup) (name . "Cleanup"))))))
  (setq-local torch-dojo-scenario-index 0
              torch-dojo-scenario-client nil
              torch-dojo-scenario-kernel-id nil
              torch-dojo-scenario-busy nil
              torch-dojo-scenario-failed nil
              torch-dojo-scenario-last-error nil
              torch-dojo-scenario-history nil
              torch-dojo-scenario-output nil
              torch-dojo-scenario-submitted nil
              torch-dojo-scenario-stale nil
              torch-dojo-scenario-view-index nil)
  (torch-dojo-scenario-status "Ready")
  (torch-dojo-scenario-focus)
  (current-buffer))

(defun torch-dojo-scenario-status (message)
  "Record MESSAGE for the bridge and minibuffer."
  (setq torch-dojo-scenario-status-text message)
  (message "%s — %s" message
           (cond (torch-dojo-scenario-failed "use r to restart or q to clean up")
                 ((>= torch-dojo-scenario-index (length torch-dojo-scenario-steps))
                  "press q to exit or r to restart")
                 (t (format "press n for %s" (torch-dojo-scenario-step-name torch-dojo-scenario-index))))))

(defun torch-dojo-scenario-focus (&optional index)
  "Move to the cell at INDEX, if that step has a source block."
  (let* ((number (or index torch-dojo-scenario-index))
         (step (and (< number (length torch-dojo-scenario-steps))
                    (torch-dojo-scenario-step number)))
         (id (and step (alist-get 'cell_id step))))
    (when id (goto-char (alist-get 'contents-begin (torch-dojo-scenario-org-cell id))))))

(defun torch-dojo-scenario-show-step-result (index result)
  "Write RESULT in the Org run copy for a cell INDEX."
  (when-let ((id (alist-get 'cell_id (torch-dojo-scenario-step index))))
    (torch-dojo-scenario-org-write-result id result)))

(defun torch-dojo-scenario-setup ()
  "Check and describe the selected rootfs and authenticated server."
  (let* ((token-path "/workspace/torch_dojo/target/insula/shared/token")
         (_ (torch-dojo-scenario-token))
         (specs (alist-get 'kernelspecs
                           (torch-dojo-scenario-api "/api/kernelspecs")))
         (selected (with-temp-buffer
                     (insert-file-contents
                      "/workspace/torch_dojo/target/insula/rootfs/selected.json")
                     (json-parse-buffer :object-type 'alist)))
         (rootfs-id (alist-get 'id selected)))
    (unless (assoc 'torch-dojo-gpu specs)
      (error "torch-dojo-gpu kernel spec unavailable"))
    (format "rootfs=%s\nrepo=/workspace/torch_dojo (read-write)\nserver=http://127.0.0.1:%s/\ntoken=%s mode=0600\nkernel=torch-dojo-gpu available"
            rootfs-id (or (getenv "DOJO_VERIFY_PORT") "8899") token-path)))

(defun torch-dojo-scenario-token ()
  "Read a secure Jupyter token without displaying its contents."
  (let ((path "/workspace/torch_dojo/target/insula/shared/token"))
    (unless (and (file-exists-p path) (= (file-modes path) #o600))
      (error "Jupyter token file missing or insecure"))
    (let ((value (string-trim (with-temp-buffer
                                (insert-file-contents path)
                                (buffer-string)))))
      (when (string-empty-p value)
        (error "Jupyter token file empty"))
      value)))

(defun torch-dojo-scenario-api (path &optional method)
  "Return parsed authenticated Jupyter API response at PATH using METHOD."
  (let* ((url-request-method (or method "GET"))
         (url-request-extra-headers
          `(("Authorization" . ,(concat "token " (torch-dojo-scenario-token)))))
         (port (or (getenv "DOJO_VERIFY_PORT") "8899"))
         (buffer (url-retrieve-synchronously
                  (format "http://127.0.0.1:%s%s" port path) t t 5)))
    (unless buffer (error "Jupyter API unavailable at port %s" port))
    (unwind-protect
        (with-current-buffer buffer
          (unless (and (boundp 'url-http-response-status)
                       (< url-http-response-status 400))
            (error "Jupyter API request failed"))
          (goto-char (point-min))
          (when (re-search-forward "\r?\n\r?\n" nil t)
            (if (looking-at-p "[[:space:]]*\\'") nil
              (json-parse-buffer :object-type 'alist :array-type 'list))))
      (kill-buffer buffer))))

(defun torch-dojo-scenario-next ()
  "Execute exactly one pending scenario step."
  (interactive)
  (when (or torch-dojo-scenario-busy torch-dojo-scenario-failed)
    (user-error "Scenario is busy or paused; use r to restart"))
  (when (>= torch-dojo-scenario-index (length torch-dojo-scenario-steps))
    (user-error "Scenario complete; use r or q"))
  (let ((index torch-dojo-scenario-index))
    (pcase (torch-dojo-scenario-step-kind index)
      ('setup (condition-case err
                  (torch-dojo-scenario-complete index (torch-dojo-scenario-setup))
                (error (torch-dojo-scenario-fail index 'error (error-message-string err)))))
      ('connect (torch-dojo-scenario-dispatch-connect))
      ('cell (torch-dojo-scenario-dispatch-cell))
      ('final (torch-dojo-scenario-complete
               index (format "completed-cells=%d\nrun-file=%s\nkernel-id=%s"
                             (- (length torch-dojo-scenario-steps) 4)
                             (or (buffer-file-name) "") torch-dojo-scenario-kernel-id)))
      ('cleanup (torch-dojo-scenario-dispatch-cleanup)))))

(defun torch-dojo-scenario-kernel-gone-p (id)
  "Return non-nil if kernel ID is absent from the authenticated server."
  (not (cl-find-if (lambda (kernel)
                     (equal id (alist-get 'id kernel)))
                   (torch-dojo-scenario-api "/api/kernels"))))

(defun torch-dojo-scenario-dispatch-cleanup ()
  "Shut down the scenario kernel and confirm it left the server."
  (let ((buffer (current-buffer))
        (id torch-dojo-scenario-kernel-id))
    (setq torch-dojo-scenario-busy t)
    (torch-dojo-scenario-transport-shutdown
     torch-dojo-scenario-client
     (lambda (ok detail)
       (with-current-buffer buffer
         (if (not ok)
             (torch-dojo-scenario-fail (1- (length torch-dojo-scenario-steps)) 'cleanup-error detail)
           (condition-case err
               (if (torch-dojo-scenario-kernel-gone-p id)
                   (progn
                     (setq torch-dojo-scenario-client nil)
                     (torch-dojo-scenario-complete
                      (1- (length torch-dojo-scenario-steps)) (format "kernel-id=%s absent from server" id)))
                 (torch-dojo-scenario-fail (1- (length torch-dojo-scenario-steps)) 'cleanup-error
                                           "kernel remains on server"))
             (error (torch-dojo-scenario-fail
                     (1- (length torch-dojo-scenario-steps)) 'cleanup-error (error-message-string err))))))))))

(defun torch-dojo-scenario-transport-shutdown (client on-done)
  "Shut down owned CLIENT and report success to ON-DONE."
  (condition-case err
      (progn
        (when client (jupyter-shutdown-kernel client))
        (funcall on-done t nil))
    (error (funcall on-done nil (error-message-string err)))))

(defun torch-dojo-scenario-clear-results ()
  (torch-dojo-scenario-org-clear-results))

(defun torch-dojo-scenario-restart ()
  "Shut down the owned kernel and restart from setup with cell edits intact."
  (interactive)
  (when torch-dojo-scenario-busy
    (user-error "Wait for the current request to finish"))
  (let ((buffer (current-buffer)))
    (torch-dojo-scenario-transport-shutdown
     torch-dojo-scenario-client
     (lambda (ok detail)
       (with-current-buffer buffer
         (if (not ok)
             (progn
               (setq torch-dojo-scenario-failed t
                     torch-dojo-scenario-last-error
                     (format "cleanup error: %s" detail))
               (write-region "kernel shutdown failed\n" nil
                             (expand-file-name "cleanup-error" (torch-dojo-scenario-dir))
                             nil 'silent)
               (torch-dojo-scenario-status torch-dojo-scenario-last-error)
               (run-hooks 'torch-dojo-scenario-change-hook))
           (let ((marker (expand-file-name "cleanup-error" (torch-dojo-scenario-dir))))
             (when (file-exists-p marker) (delete-file marker)))
           (setq torch-dojo-scenario-client nil
                 torch-dojo-scenario-kernel-id nil
                 torch-dojo-scenario-index 0
                 torch-dojo-scenario-failed nil
                 torch-dojo-scenario-last-error nil
                 torch-dojo-scenario-history nil
                 torch-dojo-scenario-output nil
                 torch-dojo-scenario-submitted nil
                 torch-dojo-scenario-stale nil
                 torch-dojo-scenario-view-index nil)
           (torch-dojo-scenario-clear-results)
           (torch-dojo-scenario-status "Ready")
           (torch-dojo-scenario-focus)
           (run-hooks 'torch-dojo-scenario-change-hook)))))))

(defun torch-dojo-scenario-exit (status)
  "Exit scenario Emacs with STATUS."
  (save-buffer)
  (run-hooks 'torch-dojo-scenario-before-exit-hook)
  (kill-emacs status))

(defun torch-dojo-scenario-quit ()
  "Clean up the owned kernel and exit this scenario Emacs process."
  (interactive)
  (when torch-dojo-scenario-busy
    (user-error "Wait for the current request to finish"))
  (let ((buffer (current-buffer))
        (failed torch-dojo-scenario-failed))
    (torch-dojo-scenario-transport-shutdown
     torch-dojo-scenario-client
     (lambda (ok detail)
       (with-current-buffer buffer
         (unless ok
           (torch-dojo-scenario-status (format "cleanup error: %s" detail))
           (write-region "kernel shutdown failed\n" nil
                         (expand-file-name "cleanup-error" (torch-dojo-scenario-dir))
                         nil 'silent))
         (when ok
           (let ((marker (expand-file-name "cleanup-error" (torch-dojo-scenario-dir))))
             (when (file-exists-p marker) (delete-file marker))))
         (torch-dojo-scenario-exit (if (or failed (not ok)) 1 0)))))))

(defun torch-dojo-scenario-complete (index result)
  "Record RESULT for completed step INDEX and move to the next step."
  (setq torch-dojo-scenario-busy nil)
  (torch-dojo-scenario-show-step-result index result)
  (push (cons index result) torch-dojo-scenario-history)
  (setq torch-dojo-scenario-index (1+ index))
  (setq torch-dojo-scenario-view-index nil)
  (torch-dojo-scenario-status
   (if (< torch-dojo-scenario-index (length torch-dojo-scenario-steps))
       (format "%s complete" (torch-dojo-scenario-step-name index))
     "Cleanup complete"))
  (torch-dojo-scenario-focus)
  (run-hooks 'torch-dojo-scenario-change-hook))

(defun torch-dojo-scenario-fail (index kind detail)
  "Pause step INDEX with failure KIND and DETAIL."
  (when (eq (alist-get 'kind (torch-dojo-scenario-step index)) 'cell)
    (push (cons index (list :error detail)) torch-dojo-scenario-output))
  (setq torch-dojo-scenario-busy nil
        torch-dojo-scenario-failed t
        torch-dojo-scenario-last-error (format "%s: %s" kind detail))
  (torch-dojo-scenario-show-step-result index
                                   (format "%s: %s" kind detail))
  (torch-dojo-scenario-status
   (format "%s — inspect the result before restarting" kind))
  (torch-dojo-scenario-focus index)
  (run-hooks 'torch-dojo-scenario-change-hook))

(defun torch-dojo-scenario-dispatch-connect ()
  "Connect from this Emacs insula to one owned kernel."
  (let ((buffer (current-buffer)))
    (setq torch-dojo-scenario-busy t)
    (condition-case err
        (torch-dojo-scenario-transport-connect
         (lambda (client id)
           (with-current-buffer buffer
             (setq torch-dojo-scenario-client client
                   torch-dojo-scenario-kernel-id id)
             (torch-dojo-scenario-complete 1 (format "kernel-id=%s" id))))
         (lambda (kind detail)
           (with-current-buffer buffer
             (torch-dojo-scenario-fail 1 kind detail))))
      (error (torch-dojo-scenario-fail 1 'error (error-message-string err))))))

(defun torch-dojo-scenario-source (index)
  "Return source for cell step INDEX after validating Org structure."
  (torch-dojo-scenario-org-source
   (alist-get 'cell_id (torch-dojo-scenario-step index))))

(defun torch-dojo-scenario-format-output (result)
  "Format a transport RESULT for the terminal Org buffer."
  (string-join
   (delq nil (list (plist-get result :stdout)
                   (plist-get result :text)
                   (when-let ((mimes (plist-get result :mime)))
                     (format "MIME: %s" (string-join mimes ", ")))))
   "\n"))

(defun torch-dojo-scenario-dispatch-cell ()
  "Execute the current editable cell once using the owned client."
  (let* ((buffer (current-buffer))
         (index torch-dojo-scenario-index)
         (code (torch-dojo-scenario-source index)))
    (setf (alist-get (alist-get 'cell_id (torch-dojo-scenario-step index))
                     torch-dojo-scenario-submitted nil nil #'equal) code)
    (setq torch-dojo-scenario-busy t)
    (condition-case err
        (torch-dojo-scenario-transport-execute
         torch-dojo-scenario-client code
         (lambda (result)
           (with-current-buffer buffer
             (push (cons index result) torch-dojo-scenario-output)
             (torch-dojo-scenario-complete
              index (torch-dojo-scenario-format-output result))))
         (lambda (kind detail)
           (with-current-buffer buffer
             (torch-dojo-scenario-fail index kind detail))))
      (error (torch-dojo-scenario-fail index 'error (error-message-string err))))))

(defun torch-dojo-scenario-load-jupyter ()
  "Load rootfs Doom's pinned emacs-jupyter packages."
  (let ((build (format "/home/dojo/.emacs.d/.local/straight/build-%d.%d"
                       emacs-major-version emacs-minor-version)))
    (dolist (dir (directory-files build t directory-files-no-dot-files-regexp))
      (when (file-directory-p dir) (add-to-list 'load-path dir))))
  (require 'jupyter)
  (require 'jupyter-server)
  (require 'jupyter-client)
  (require 'jupyter-messages))

(defun torch-dojo-scenario-transport-connect (on-success on-error)
  "Start one emacs-jupyter client, calling ON-SUCCESS or ON-ERROR."
  (condition-case err
      (let* ((token (or (getenv "DOJO_VERIFY_TOKEN")
                        (torch-dojo-scenario-token)))
             (port (or (getenv "DOJO_VERIFY_PORT") "8899"))
             (url (format "http://127.0.0.1:%s" port)))
        (torch-dojo-scenario-load-jupyter)
        (let* ((auth `(("Authorization" . ,(concat "token " token))))
               (jupyter-api-authentication-method auth)
               (server (jupyter-server :url url :auth auth))
               (client (jupyter-client
                        (jupyter-kernel :server server :spec "torch-dojo-gpu")
                        'jupyter-kernel-client))
               (id (jupyter-kernel-action client #'jupyter-server-kernel-id)))
          (funcall on-success client id)))
    (error (funcall on-error 'error (error-message-string err)))))

(defun torch-dojo-scenario-transport-execute (client code on-success on-error)
  "Execute CODE once with CLIENT and deliver output via callbacks."
  (condition-case err
      (let ((messages
             (jupyter-run-with-client client
               (jupyter-messages
                (jupyter-execute-request :code code :store-history nil
                                         :handlers nil :allow-stdin nil)
                60)))
            (stdout nil) (plain nil) (mime nil) (failure nil))
        (dolist (msg messages)
          (pcase (jupyter-message-type msg)
            ("stream" (push (jupyter-message-get msg :text) stdout))
            ((or "execute_result" "display_data")
             (when-let ((text (jupyter-message-data msg :text/plain)))
               (push text plain))
             (let ((data (jupyter-message-get msg :data)))
               (while data
                 (let ((key (pop data)))
                   (pop data)
                   (unless (eq key :text/plain)
                     (push (substring (symbol-name key) 1) mime))))))
            ("error" (setq failure (string-join
                                     (jupyter-message-get msg :traceback) "\n")))))
        (if failure
            (funcall on-error 'error failure)
          (funcall on-success (list :stdout (string-join (nreverse stdout) "")
                                    :text (string-join (nreverse plain) "\n")
                                    :mime (delete-dups (nreverse mime))))))
    (jupyter-timeout-before-idle (funcall on-error 'unknown "outcome unknown"))
    (error (funcall on-error 'error (error-message-string err)))))

(defun torch-dojo-scenario-previous ()
  "Move back through completed steps without executing them."
  (interactive)
  (when (> torch-dojo-scenario-index 0)
    (torch-dojo-scenario-view-completed
     (max 0 (if torch-dojo-scenario-view-index
                (1- torch-dojo-scenario-view-index)
              (1- torch-dojo-scenario-index))))))

(provide 'scenario)
;;; scenario.el ends here
