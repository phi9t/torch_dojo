;;; scenario_bridge.el --- Private scenario RPC -*- lexical-binding: t; -*-
(require 'scenario)
(require 'json)
(require 'cl-lib)
(require 'subr-x)

(defconst torch-dojo-scenario-bridge-limit (* 1024 1024))
(defvar torch-dojo-scenario-bridge-buffer nil)
(defvar torch-dojo-scenario-bridge-server nil)
(defvar torch-dojo-scenario-bridge-pending nil)

(defun torch-dojo-scenario-bridge-set-source (id source)
  "Replace cell ID's Python source and save the run copy."
  (unless (and (stringp source)
               (<= (string-bytes source) torch-dojo-scenario-bridge-limit))
    (error "Invalid source block"))
  (torch-dojo-scenario-set-source id source))

(defun torch-dojo-scenario-bridge-redact (message)
  (let ((path "/workspace/torch_dojo/target/insula/shared/token"))
    (if (not (file-readable-p path)) message
      (let ((token (string-trim (with-temp-buffer
                                  (insert-file-contents path)
                                  (buffer-string)))))
        (if (string-empty-p token) message
          (replace-regexp-in-string (regexp-quote token) "[redacted]" message t t))))))

(defun torch-dojo-scenario-bridge-response (id &optional error)
  `((id . ,(or id :null))
    (ok . ,(if error :false t))
    ,@(if error `((error . ,(torch-dojo-scenario-bridge-redact error)))
        `((snapshot . ,(torch-dojo-scenario-snapshot))))))

(defun torch-dojo-scenario-bridge-dispatch (request)
  "Dispatch one allowlisted REQUEST in the scenario buffer."
  (let ((id (alist-get 'id request))
        (command (alist-get 'command request))
        (args (alist-get 'args request)))
    (condition-case err
        (progn
          (unless (and (integerp id) (stringp command))
            (error "Invalid request"))
          (pcase command
            ("snapshot" nil)
            ("set_source" (torch-dojo-scenario-bridge-set-source
                           (alist-get 'cell_id args) (alist-get 'source args)))
            ("next" (torch-dojo-scenario-next))
            ("previous"
             (let ((target (alist-get 'index args)))
               (if (null target)
                   (torch-dojo-scenario-previous)
                 (torch-dojo-scenario-view-completed target))))
            ("restart" (torch-dojo-scenario-restart))
            ("quit" (torch-dojo-scenario-quit))
            (_ (error "Unknown command")))
          (torch-dojo-scenario-bridge-response id))
      (error (torch-dojo-scenario-bridge-response id (error-message-string err))))))

(defun torch-dojo-scenario-bridge-send (process response)
  (when (process-live-p process)
    (process-send-string process (concat (json-serialize response) "\n"))))

(defun torch-dojo-scenario-bridge-completed ()
  (when torch-dojo-scenario-bridge-pending
    (pcase-let ((`(,process . ,id) torch-dojo-scenario-bridge-pending))
      (setq torch-dojo-scenario-bridge-pending nil)
      (torch-dojo-scenario-bridge-send process (torch-dojo-scenario-bridge-response id)))))

(defun torch-dojo-scenario-bridge-handle-line (process line)
  (with-current-buffer torch-dojo-scenario-bridge-buffer
    (condition-case err
        (let* ((request (condition-case nil
                            (json-parse-string line :object-type 'alist)
                          (error (error "Malformed request"))))
               (id (alist-get 'id request))
               (command (alist-get 'command request)))
          (if (and torch-dojo-scenario-bridge-pending
                   (not (equal command "snapshot")))
              (torch-dojo-scenario-bridge-send
               process (torch-dojo-scenario-bridge-response
                        id "Scenario request in progress"))
            (when (member command '("next" "restart" "quit"))
              (setq torch-dojo-scenario-bridge-pending (cons process id)))
            (let ((response (torch-dojo-scenario-bridge-dispatch request)))
              (when torch-dojo-scenario-bridge-pending
                (unless torch-dojo-scenario-busy
                  (setq torch-dojo-scenario-bridge-pending nil)
                  (torch-dojo-scenario-bridge-send process response)))
              (unless (member command '("next" "restart" "quit"))
                (torch-dojo-scenario-bridge-send process response)))))
      (error
       (torch-dojo-scenario-bridge-send
        process (torch-dojo-scenario-bridge-response nil (error-message-string err)))))))

(defun torch-dojo-scenario-bridge-filter (process chunk)
  (let ((data (concat (or (process-get process 'partial) "") chunk)))
    (if (> (string-bytes data) torch-dojo-scenario-bridge-limit)
        (progn
          (process-put process 'partial "")
          (torch-dojo-scenario-bridge-send
           process (torch-dojo-scenario-bridge-response nil "Request too large")))
      (while (string-match "\n" data)
        (let ((line (substring data 0 (match-beginning 0))))
          (setq data (substring data (match-end 0)))
          (torch-dojo-scenario-bridge-handle-line process line)))
      (process-put process 'partial data))))

(defun torch-dojo-scenario-bridge-start ()
  "Open scenario and start its current-user-only Unix socket server."
  (setq torch-dojo-scenario-bridge-buffer (torch-dojo-scenario-open))
  (with-current-buffer torch-dojo-scenario-bridge-buffer
    (add-hook 'torch-dojo-scenario-change-hook
              #'torch-dojo-scenario-bridge-completed nil t)
    (add-hook 'torch-dojo-scenario-before-exit-hook
              #'torch-dojo-scenario-bridge-completed nil t))
  (let* ((path (expand-file-name "bridge.sock" (torch-dojo-scenario-dir))))
    (when (file-exists-p path) (delete-file path))
    (setq torch-dojo-scenario-bridge-server
          (make-network-process :name "torch-dojo-scenario-bridge"
                                :family 'local :server t :service path
                                :filter #'torch-dojo-scenario-bridge-filter))
    (set-file-modes path #o600)
    torch-dojo-scenario-bridge-server))

(defun torch-dojo-scenario-bridge-serve ()
  "Serve the bridge until the scenario quits."
  (torch-dojo-scenario-bridge-start)
  (while t (accept-process-output nil 1)))

(provide 'scenario_bridge)
;;; scenario_bridge.el ends here
