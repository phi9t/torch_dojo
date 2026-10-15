;;; verify_attach.el -*- lexical-binding: t; -*-
(require 'cl-lib)
(require 'subr-x)
(let* ((token-file "/workspace/torch_dojo/target/insula/shared/token")
       (token (or (getenv "DOJO_VERIFY_TOKEN") (string-trim (with-temp-buffer (insert-file-contents token-file) (buffer-string))))))
  (unless (and (= (file-modes token-file) #o600) (not (string-empty-p token)))
    (error "token missing or insecure"))
  (setq user-emacs-directory (expand-file-name "~/.emacs.d/"))
  (let ((build (format "~/.emacs.d/.local/straight/build-%d.%d" emacs-major-version emacs-minor-version)))
    (dolist (dir (directory-files build t directory-files-no-dot-files-regexp))
      (when (file-directory-p dir) (add-to-list 'load-path dir))))
  (require 'jupyter)
  (require 'jupyter-server)
  (require 'jupyter-client)
  (require 'jupyter-messages)
  (let* ((url (concat "http://127.0.0.1:" (or (getenv "DOJO_VERIFY_PORT") "8899")))
         (server (jupyter-server :url url :auth `(("Authorization" . ,(concat "token " token)))))
         (jupyter-api-authentication-method `(("Authorization" . ,(concat "token " token))))
         (client (jupyter-client (jupyter-kernel :server server :spec "torch-dojo-gpu") 'jupyter-kernel-client)))
    (unwind-protect
        (let ((result (jupyter-run-with-client client
                        (jupyter-mlet* ((reply (jupyter-result
                                                  (jupyter-message-subscribed
                                                   (jupyter-execute-request
                                                    :code "'torch-dojo-attach-ok'"
                                                    :store-history nil :handlers nil)
                                                   `(("execute_reply"
                                                      ,(jupyter-message-lambda (status evalue)
                                                         (unless (equal status "ok")
                                                           (error "%s" evalue))))))
                                                  60)))
                          (jupyter-return (jupyter-message-data reply :text/plain))))))
          (unless (string-match-p "torch-dojo-attach-ok" (format "%S" result))
            (error "kernel execution evidence missing: %S" result))
          (princ "Emacs attached and executed in torch-dojo-gpu\n"))
      (ignore-errors (jupyter-shutdown-kernel client)))))
