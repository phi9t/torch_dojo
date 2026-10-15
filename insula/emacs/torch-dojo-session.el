;;; torch-dojo-session.el --- Persistent server REPL transport -*- lexical-binding: t; -*-

;;; Commentary:
;; Scoped transport for Dojo clients. Evaluation and rendering remain in
;; emacs-jupyter. Reconnection reuses the frontend identity; sends are never
;; retried. This module targets the pinned checkout documented in session.md.

;;; Code:
(require 'cl-lib)
(require 'jupyter)
(require 'jupyter-server)
(require 'jupyter-messages)
(require 'websocket)

(defgroup torch-dojo-session nil "Persistent Dojo Jupyter sessions." :group 'jupyter)
(defcustom torch-dojo-session-terminal-history-limit 128
  "Maximum terminal execution outcomes retained per client.
Incomplete outcomes are never evicted. Lookup returns nil after eviction."
  :type 'natnum :group 'torch-dojo-session)

(define-error 'torch-dojo-session-unavailable "Dojo session is not ready")

(cl-defstruct (torch-dojo-session-kernel (:include jupyter-server-kernel))
  (frontend (jupyter-session))
  socket
  (state 'disconnected)
  generation
  verified-generation
  probe-id
  probe-request
  client
  order
  (fragment "")
  (outcomes (make-hash-table :test #'equal)))

(defun torch-dojo-session--kernel (client)
  "Return CLIENT's Dojo kernel or signal an error."
  (let ((kernel (jupyter-kernel-action client #'identity)))
    (unless (torch-dojo-session-kernel-p kernel)
      (user-error "Client does not use the Dojo session transport"))
    kernel))

(defun torch-dojo-session--prune (kernel)
  "Bound KERNEL's terminal history, preserving incomplete outcomes."
  (let ((count 0) retained)
    (dolist (id (torch-dojo-session-kernel-order kernel))
      (let ((state (plist-get (gethash id (torch-dojo-session-kernel-outcomes kernel)) :state)))
        (if (and (memq state '(complete error state-lost))
                 (>= count torch-dojo-session-terminal-history-limit))
            (remhash id (torch-dojo-session-kernel-outcomes kernel))
          (when (memq state '(complete error state-lost)) (cl-incf count))
          (push id retained))))
    (setf (torch-dojo-session-kernel-order kernel) (nreverse retained))))

(defun torch-dojo-session--lost (kernel)
  "Mark KERNEL disconnected and incomplete evaluations unknown."
  (unless (eq (torch-dojo-session-kernel-state kernel) 'state-lost)
    (setf (torch-dojo-session-kernel-state kernel) 'disconnected))
  (maphash (lambda (_id outcome)
             (when (eq (plist-get outcome :state) 'pending)
               (plist-put outcome :state 'unknown)))
           (torch-dojo-session-kernel-outcomes kernel)))

(defun torch-dojo-session--observe (kernel msg)
  "Update KERNEL's generation and submission outcomes from MSG."
  (let* ((generation (plist-get (plist-get msg :header) :session))
         (old (torch-dojo-session-kernel-generation kernel))
         (type (plist-get (plist-get msg :header) :msg_type))
         (id (plist-get (plist-get msg :parent_header) :msg_id))
         (content (plist-get msg :content))
         (outcome (gethash id (torch-dojo-session-kernel-outcomes kernel))))
    (when generation
      (when (and old (not (equal old generation)))
        (setf (torch-dojo-session-kernel-state kernel) 'state-lost)
        (maphash (lambda (_key value)
                   (when (memq (plist-get value :state) '(pending unknown))
                     (plist-put value :state 'state-lost)))
                 (torch-dojo-session-kernel-outcomes kernel)))
      (setf (torch-dojo-session-kernel-generation kernel) generation))
    (when (and (equal type "kernel_info_reply")
               (equal id (torch-dojo-session-kernel-probe-id kernel)))
      (setf (torch-dojo-session-kernel-verified-generation kernel) generation)
      (when (eq (torch-dojo-session-kernel-state kernel) 'checking)
        (setf (torch-dojo-session-kernel-state kernel) 'connected)))
    (when (and outcome (not (eq (plist-get outcome :state) 'state-lost)))
      (pcase type
        ("execute_reply" (plist-put outcome :reply (plist-get content :status)))
        ("status" (when (equal (plist-get content :execution_state) "idle")
                    (plist-put outcome :idle t))))
      (when (and (plist-get outcome :reply) (plist-get outcome :idle))
        (plist-put outcome :state
                   (if (equal (plist-get outcome :reply) "ok") 'complete 'error))))
    (torch-dojo-session--prune kernel)))

(defun torch-dojo-session--close (kernel)
  "Close only KERNEL's frontend socket, retaining kernel and request identity."
  (torch-dojo-session--lost kernel)
  (when-let ((ws (torch-dojo-session-kernel-socket kernel)))
    (websocket-close ws))
  (setf (torch-dojo-session-kernel-socket kernel) nil
        (torch-dojo-session-kernel-fragment kernel) ""))

(defun torch-dojo-session--send (kernel channel type content id)
  "Send one message through KERNEL, never retrying after uncertain delivery."
  (unless (and (or (eq (torch-dojo-session-kernel-state kernel) 'connected)
                   (and (not (equal type "execute_request"))
                        (memq (torch-dojo-session-kernel-state kernel) '(checking state-lost))))
               (websocket-openp (torch-dojo-session-kernel-socket kernel)))
    (signal 'torch-dojo-session-unavailable
            '("Reconnect explicitly; acknowledge state loss after a restart")))
  (when (and (equal type "kernel_info_request")
             (memq (torch-dojo-session-kernel-state kernel) '(checking state-lost)))
    (setf (torch-dojo-session-kernel-probe-id kernel) id))
  (when (equal type "execute_request")
    (push id (torch-dojo-session-kernel-order kernel))
    (puthash id (list :state 'pending :reply nil :idle nil :generation (torch-dojo-session-kernel-generation kernel))
             (torch-dojo-session-kernel-outcomes kernel)))
  (condition-case err
      (websocket-send-text
       (torch-dojo-session-kernel-socket kernel)
       (jupyter-encode-raw-message (torch-dojo-session-kernel-frontend kernel) type
         :channel channel :msg-id id :content content))
    (error
     (torch-dojo-session--lost kernel)
     ;; Delivery may have happened. Keep the original request as unknown.
     (signal 'torch-dojo-session-unavailable
             (list "Send failed; execution outcome unknown; no retry" (car err))))))

(defun torch-dojo-session--open (kernel publisher)
  "Open KERNEL's socket with its existing frontend identity and PUBLISHER."
  (when (and (eq (torch-dojo-session-kernel-state kernel) 'disconnected)
             (torch-dojo-session-kernel-socket kernel))
    (torch-dojo-session--close kernel))
  (unless (websocket-openp (torch-dojo-session-kernel-socket kernel))
    (setf (torch-dojo-session-kernel-fragment kernel) ""
          (torch-dojo-session-kernel-verified-generation kernel) nil
          (torch-dojo-session-kernel-probe-id kernel) nil
          (torch-dojo-session-kernel-probe-request kernel) nil)
    (let* ((server (jupyter-server-kernel-server kernel))
           (session (torch-dojo-session-kernel-frontend kernel))
           (ws (jupyter-api/kernels
                server "WS" (jupyter-server-kernel-id kernel) "channels"
                `(("session_id" . ,(jupyter-session-id session)))
                :custom-header-alist (jupyter-api-auth-headers server)
                :on-open (lambda (_ws)
                           (unless (eq (torch-dojo-session-kernel-state kernel) 'state-lost)
                             (setf (torch-dojo-session-kernel-state kernel)
                                   (if (torch-dojo-session-kernel-generation kernel) 'checking 'connected))))
                :on-close (lambda (socket)
                            (when (eq socket (torch-dojo-session-kernel-socket kernel))
                              (torch-dojo-session--lost kernel)))
                :on-error (lambda (socket _type _err)
                            (when (eq socket (torch-dojo-session-kernel-socket kernel))
                              (torch-dojo-session--lost kernel)))
                :on-message
                (lambda (socket frame)
                  (when (eq socket (torch-dojo-session-kernel-socket kernel))
                    (setf (torch-dojo-session-kernel-fragment kernel)
                          (concat (torch-dojo-session-kernel-fragment kernel)
                                  (websocket-frame-payload frame)))
                    (when (websocket-frame-completep frame)
                      (let ((msg (jupyter-read-plist-from-string
                                  (torch-dojo-session-kernel-fragment kernel))))
                        (setf (torch-dojo-session-kernel-fragment kernel) "")
                        (torch-dojo-session--observe kernel msg)
                        (jupyter-run-with-io publisher
                          (jupyter-publish (cons 'message msg))))))))))
      (setf (websocket-client-data ws)
            (list :id (jupyter-server-kernel-id kernel) :session session)
            (torch-dojo-session-kernel-socket kernel) ws)
      ;; websocket-open returns before the HTTP upgrade callback. Wait only
      ;; for the handshake, so the first client request has a ready transport.
      (let ((deadline (+ (float-time) 10)))
        (while (and (eq (websocket-ready-state ws) 'connecting)
                    (< (float-time) deadline))
          (accept-process-output (websocket-conn ws) 0.05))
        (unless (eq (websocket-ready-state ws) 'open)
          (torch-dojo-session--close kernel)
          (signal 'torch-dojo-session-unavailable '("WebSocket handshake failed"))))
      (when (torch-dojo-session-kernel-generation kernel)
        (setf (torch-dojo-session-kernel-probe-request kernel)
              (jupyter-run-with-client (torch-dojo-session-kernel-client kernel)
                (jupyter-sent (jupyter-kernel-info-request))))))))

(cl-defmethod jupyter-client :around ((kernel torch-dojo-session-kernel) &optional _class)
  "Remember KERNEL's client for generation probes on every reopening path."
  (let ((client (cl-call-next-method)))
    (setf (torch-dojo-session-kernel-client kernel) client)
    client))

(cl-defmethod jupyter-io ((kernel torch-dojo-session-kernel))
  "Use a Dojo-owned transport for KERNEL without altering other clients."
  (jupyter-launch kernel)
  (letrec ((publisher
            (jupyter-publisher
              (lambda (event)
                (pcase event
                  (`(message . ,rest) (jupyter-content rest))
                  (`(send ,channel ,type ,content ,id)
                   (torch-dojo-session--send kernel channel type content id))
                  ('start (torch-dojo-session--open kernel publisher))
                  ('stop (torch-dojo-session--close kernel)))))))
    (torch-dojo-session--open kernel publisher)
    (list publisher
          (jupyter-subscriber
            (lambda (action)
              (pcase action
                ('interrupt (jupyter-interrupt kernel))
                ((or 'connect '(connect)) (torch-dojo-session--open kernel publisher))
                ((or 'disconnect '(disconnect)) (torch-dojo-session--close kernel))
                ('shutdown (jupyter-shutdown kernel) (torch-dojo-session--close kernel))
                ('restart
                 ;; Block new code before the REST call can replace the kernel.
                 (setf (torch-dojo-session-kernel-state kernel) 'state-lost
                       (torch-dojo-session-kernel-verified-generation kernel) nil)
                 (jupyter-restart kernel)
                 (torch-dojo-session--close kernel)
                 (torch-dojo-session--open kernel publisher))
                (`(action ,fn) (funcall fn kernel))))))))

;;;###autoload
(defun torch-dojo-connect-server-repl
    (server kernel-id &optional repl-name associate-buffer client-class display)
  "Attach a Dojo REPL to SERVER's existing KERNEL-ID.
Remaining arguments match `jupyter-connect-server-repl'."
  (interactive
   (let ((server (jupyter-current-server current-prefix-arg)))
     (list server (completing-read "Kernel ID: "
                                  (mapcar (lambda (k) (plist-get k :id))
                                          (jupyter-api-get-kernel server)))
           nil t nil t)))
  (jupyter-server-repl
   (make-torch-dojo-session-kernel :server server :id kernel-id)
   repl-name associate-buffer client-class display))

;;;###autoload
(defun torch-dojo-run-server-repl
    (server kernel-name &optional repl-name associate-buffer client-class display)
  "Start KERNEL-NAME on SERVER and attach a Dojo REPL."
  (interactive
   (let ((server (jupyter-current-server current-prefix-arg)))
     (list server (jupyter-completing-read-kernelspec (jupyter-kernelspecs server))
           nil t nil t)))
  (when (jupyter-kernelspec-p kernel-name)
    (setq kernel-name (jupyter-kernelspec-name kernel-name)))
  (let ((id (plist-get (jupyter-api-start-kernel server kernel-name) :id)))
    (condition-case err
        (torch-dojo-connect-server-repl server id repl-name associate-buffer client-class display)
      (error (jupyter-api-shutdown-kernel server id)
             (signal (car err) (cdr err))))))

(defun torch-dojo-session-disconnect (&optional client)
  "Detach CLIENT (or the current client) without shutting down its kernel."
  (interactive)
  (torch-dojo-session--close (torch-dojo-session--kernel (or client jupyter-current-client))))

(defun torch-dojo-session-reconnect (&optional client)
  "Reconnect CLIENT's transport without resubmitting any evaluations.
Return the generation-check request. New code is rejected until its reply
arrives. Use `torch-dojo-session-status' to observe the resulting state."
  (interactive)
  (setq client (or client jupyter-current-client))
  (let ((kernel (torch-dojo-session--kernel client)))
    (torch-dojo-session--close kernel)
    (jupyter-run-with-io (jupyter-kernel-io client) (jupyter-publish 'start))
    (torch-dojo-session-kernel-probe-request kernel)))

(defun torch-dojo-session-status (&optional client)
  "Return CLIENT's connection state and distinct session identities."
  (interactive)
  (let* ((kernel (torch-dojo-session--kernel (or client jupyter-current-client)))
         (status (list :state (torch-dojo-session-kernel-state kernel)
                       :kernel-id (jupyter-server-kernel-id kernel)
                       :frontend-id (jupyter-session-id (torch-dojo-session-kernel-frontend kernel))
                       :generation (torch-dojo-session-kernel-generation kernel))))
    (when (called-interactively-p 'interactive) (message "%S" status))
    status))

(defun torch-dojo-session-request-state (client id)
  "Return CLIENT's observed execution outcome for request ID, or nil."
  (plist-get (gethash id (torch-dojo-session-kernel-outcomes (torch-dojo-session--kernel client))) :state))

(defun torch-dojo-session-acknowledge-restart (&optional client)
  "Explicitly accept CLIENT's restarted, empty workspace; never replay code."
  (interactive)
  (let ((kernel (torch-dojo-session--kernel (or client jupyter-current-client))))
    (unless (and (eq (torch-dojo-session-kernel-state kernel) 'state-lost)
                 (torch-dojo-session-kernel-verified-generation kernel)
                 (equal (torch-dojo-session-kernel-generation kernel)
                        (torch-dojo-session-kernel-verified-generation kernel))
                 (websocket-openp (torch-dojo-session-kernel-socket kernel)))
      (user-error "No connected restarted workspace to acknowledge"))
    (setf (torch-dojo-session-kernel-state kernel) 'connected)))

(provide 'torch-dojo-session)
;;; torch-dojo-session.el ends here
