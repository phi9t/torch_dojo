;;; session_test.el --- Live GPU recovery regression -*- lexical-binding: t; -*-
(require 'ert)
(require 'cl-lib)
(require 'json)
(require 'subr-x)
(let ((build (format "~/.emacs.d/.local/straight/build-%d.%d" emacs-major-version emacs-minor-version)))
  (dolist (dir (directory-files build t directory-files-no-dot-files-regexp))
    (when (file-directory-p dir) (add-to-list 'load-path dir))))
(add-to-list 'load-path "/workspace/torch_dojo/insula/emacs")
(require 'jupyter)
(require 'jupyter-server)
(require 'jupyter-messages)
(require 'torch-dojo-session nil t)
(setq jupyter-eval-use-overlays nil)
(setq jupyter-eval-short-result-display-function #'message)

(defmacro dojo-test-with-kernel (&rest body)
  `(let* ((token (string-trim (with-temp-buffer
                               (insert-file-contents "/workspace/torch_dojo/target/insula/shared/token")
                               (buffer-string))))
          (auth `(("Authorization" . ,(concat "token " token))))
          (jupyter-api-authentication-method auth)
          (server (jupyter-server :url "http://127.0.0.1:8899" :auth auth))
          (id (plist-get (jupyter-api-start-kernel server "torch-dojo-gpu") :id))
          (client nil))
     (unwind-protect
         (with-temp-buffer
           (python-mode)
           (setq client
                 (if (featurep 'torch-dojo-session)
                     (torch-dojo-connect-server-repl server id "Dojo recovery test" t nil nil)
                   (jupyter-client (jupyter-kernel :server server :id id) 'jupyter-kernel-client)))
           (setq-local jupyter-current-client client)
           ,@body)
       (when client
         (ignore-errors
           (jupyter-run-with-io (jupyter-kernel-io client) (jupyter-publish 'stop))))
       (ignore-errors (jupyter-api-shutdown-kernel server id)))))

(defun dojo-test-execute (code)
  (let ((request (jupyter-eval-string code)))
    (should (jupyter-wait-until-idle request 30))
    (should (equal (jupyter-message-get
                    (jupyter-find-message "execute_reply" (jupyter-request-messages request)) :status) "ok"))
    request))

(ert-deftest dojo-session-recovers-offline-completion-without-replay ()
  (dojo-test-with-kernel
   (dojo-test-execute "import torch, time\nmodel = torch.nn.Linear(8, 8).cuda()\nx = torch.randn(8, 8, device='cuda')\nidentity = (id(model), id(x), x.data_ptr())\ncounter = 0")
   (let ((frontend (plist-get (torch-dojo-session-status client) :frontend-id))
         (request (jupyter-eval-string
                   "counter += 1\nprint('started', flush=True)\ntime.sleep(2)\nprint('offline-result', counter, flush=True)")))
     (should (jupyter-wait-until request "stream" (lambda (m) (string-match-p "started" (jupyter-message-get m :text))) 10))
     (if (featurep 'torch-dojo-session) (torch-dojo-session-disconnect client)
       (jupyter-run-with-io (jupyter-kernel-io client) (jupyter-publish 'stop)))
     (should (eq (torch-dojo-session-request-state client (jupyter-request-id request)) 'unknown))
     (sleep-for 3)
     (if (featurep 'torch-dojo-session) (let ((info (torch-dojo-session-reconnect client)))
         (should (jupyter-wait-until info "kernel_info_reply" (lambda (_m) t) 10)))
       (jupyter-run-with-io (jupyter-kernel-io client) (jupyter-publish 'start)))
     (should (jupyter-wait-until-idle request 8))
     (should (equal frontend (plist-get (torch-dojo-session-status client) :frontend-id)))
     (should (eq (torch-dojo-session-request-state client (jupyter-request-id request)) 'complete))
     (should (cl-find-if
              (lambda (m) (and (equal (jupyter-message-type m) "stream")
                              (string-match-p "offline-result 1" (jupyter-message-get m :text))))
              (jupyter-request-messages request)))
     (dojo-test-execute "assert counter == 1\nassert identity == (id(model), id(x), x.data_ptr())\nassert model(x).is_cuda"))))

(ert-deftest dojo-session-rejects-disconnected-code-without-queueing ()
  (dojo-test-with-kernel
   (dojo-test-execute "counter = 0")
   (torch-dojo-session-disconnect client)
   (should (eq (plist-get (torch-dojo-session-status client) :state) 'disconnected))
   (should-error (jupyter-eval-string "counter += 100") :type 'torch-dojo-session-unavailable)
   (let ((info (torch-dojo-session-reconnect client)))
     (should (jupyter-wait-until info "kernel_info_reply" (lambda (_m) t) 10)))
   (dojo-test-execute "assert counter == 0")))

(ert-deftest dojo-session-detects-restart-and-does-not-replay-pending-code ()
  (dojo-test-with-kernel
   (dojo-test-execute "import time
fixture = object()")
   (let ((old (torch-dojo-session-status client))
         (pending (jupyter-eval-string "print('restart-started', flush=True)\ntime.sleep(30)\nshould_never_replay = True")))
     (should (jupyter-wait-until pending "stream" (lambda (_m) t) 10))
     (torch-dojo-session-disconnect client)
     (jupyter-api-restart-kernel server id)
     (sleep-for 1)
     (let ((info (torch-dojo-session-reconnect client)))
       (should (jupyter-wait-until info "kernel_info_reply" (lambda (_m) t) 15)))
     (let ((new (torch-dojo-session-status client)))
       (should (equal (plist-get old :kernel-id) (plist-get new :kernel-id)))
       (should (equal (plist-get old :frontend-id) (plist-get new :frontend-id)))
       (should-not (equal (plist-get old :generation) (plist-get new :generation)))
       (should (eq (plist-get new :state) 'state-lost)))
     (should (memq (torch-dojo-session-request-state client (jupyter-request-id pending)) '(state-lost error)))
     (should-error (jupyter-eval-string "wrong_workspace = True") :type 'torch-dojo-session-unavailable)
     (torch-dojo-session-acknowledge-restart client)
     (dojo-test-execute "assert 'fixture' not in globals()\nassert 'should_never_replay' not in globals()\nassert 'wrong_workspace' not in globals()"))))

(ert-deftest dojo-session-public-restart-blocks-evaluation-until-acknowledged ()
  (dojo-test-with-kernel
   (dojo-test-execute "fixture = object()")
   (jupyter-restart-kernel client)
   (should-error (jupyter-eval-string "wrong_workspace = True") :type 'torch-dojo-session-unavailable)))

(ert-deftest dojo-session-inherited-connect-verifies-generation ()
  (dojo-test-with-kernel
   (dojo-test-execute "fixture = object()")
   (jupyter-disconnect client)
   (should (eq (plist-get (torch-dojo-session-status client) :state) 'disconnected))
   (jupyter-api-restart-kernel server id)
   (jupyter-connect client)
   (should-error (jupyter-eval-string "wrong_workspace = True") :type 'torch-dojo-session-unavailable)))

(ert-deftest dojo-session-starts-the-selected-kernelspec ()
  (dojo-test-with-kernel
   (let (started)
     (unwind-protect
         (progn
           (setq started (torch-dojo-run-server-repl
                          server (jupyter-guess-kernelspec "torch-dojo-gpu" (jupyter-kernelspecs server))
                          "Dojo launch test" t nil nil))
           (setq-local jupyter-current-client started)
           (dojo-test-execute "import torch\nassert torch.ones(1, device='cuda').is_cuda"))
       (when started (jupyter-shutdown-kernel started))))))

(ert-deftest dojo-session-does-not-retry-a-send-that-fails-after-delivery ()
  (dojo-test-with-kernel
   (dojo-test-execute "counter = 0")
   (let ((send (symbol-function 'websocket-send-text)) failed-id)
     ;; Inject failure at the transport boundary *after* real delivery.
     (cl-letf (((symbol-function 'websocket-send-text)
                (lambda (ws text)
                  (let ((raw (json-parse-string text :object-type 'plist)))
                    (funcall send ws text)
                    (when (equal (plist-get (plist-get raw :header) :msg_type) "execute_request")
                      (setq failed-id (plist-get (plist-get raw :header) :msg_id))
                      (signal 'websocket-closed '("injected after delivery")))))))
       (should-error (jupyter-eval-string "counter += 1") :type 'torch-dojo-session-unavailable))
     (should failed-id)
     (should (eq (torch-dojo-session-request-state client failed-id) 'unknown))
     (let ((info (torch-dojo-session-reconnect client)))
       (should (jupyter-wait-until info "kernel_info_reply" (lambda (_m) t) 10)))
     (dojo-test-execute "assert counter == 1"))))

(ert-deftest dojo-session-bounds-terminal-outcome-history ()
  (dojo-test-with-kernel
   (let ((torch-dojo-session-terminal-history-limit 2) ids)
     (dotimes (_ 4)
       (push (jupyter-request-id (dojo-test-execute "pass")) ids))
     (should (eq (torch-dojo-session-request-state client (car ids)) 'complete))
     (should (eq (torch-dojo-session-request-state client (cadr ids)) 'complete))
     (should-not (torch-dojo-session-request-state client (nth 2 ids)))
     (should-not (torch-dojo-session-request-state client (nth 3 ids))))))

(ert-run-tests-batch-and-exit)
