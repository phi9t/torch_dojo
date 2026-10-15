;;; scenario_bridge_test.el -*- lexical-binding: t; -*-
(require 'ert)
(require 'cl-lib)
(load-file "/workspace/torch_dojo/insula/emacs/scenario.el")
(load-file "/workspace/torch_dojo/insula/emacs/scenario_bridge.el")

(defmacro dojo-bridge-with-run (&rest body)
  `(let* ((_ (make-directory "/workspace/torch_dojo/target/insula/scenario" t))
          (dir (make-temp-file "/workspace/torch_dojo/target/insula/scenario/bridge-ert-" t))
          (process-environment (cons (concat "DOJO_SCENARIO_DIR=" dir) process-environment)))
     (unwind-protect
         (progn
           (copy-file "/workspace/torch_dojo/insula/scenarios/default.org"
                      (expand-file-name "run.org" dir))
           (torch-dojo-scenario-open)
           ,@body)
       (when-let ((buf (get-file-buffer (expand-file-name "run.org" dir))))
         (kill-buffer buf))
       (delete-directory dir t))))

(ert-deftest dojo-bridge-snapshot-and-edit ()
  (dojo-bridge-with-run
   (let ((initial (torch-dojo-scenario-snapshot)))
     (should (= (alist-get 'pending_index initial) 0))
     (should (= (length (alist-get 'steps initial)) 7))
     (should (integerp (alist-get 'emacs_pid initial)))
     (should (equal (alist-get 'kind (aref (alist-get 'steps initial) 0)) "setup"))
     (should (equal (alist-get 'cell_id (aref (alist-get 'steps initial) 2)) "cell-1"))
     (should (equal (alist-get 'block_name (aref (alist-get 'steps initial) 2))
                    "context")))
   (torch-dojo-scenario-bridge-set-source "cell-1" "print('café')\nprint(2)\n")
   (should (equal (torch-dojo-scenario-source 2) "print('café')\nprint(2)\n"))
   (should-error (torch-dojo-scenario-bridge-set-source "cell-1" "#+END_SRC\n"))
   (should (equal (torch-dojo-scenario-source 2) "print('café')\nprint(2)\n"))
   (should (string-match-p "café" (with-temp-buffer
                                    (insert-file-contents (expand-file-name "run.org" dir))
                                    (buffer-string))))))

(ert-deftest dojo-bridge-allowlist-and-one-step ()
  (dojo-bridge-with-run
   (cl-letf (((symbol-function 'torch-dojo-scenario-setup) (lambda () "ready")))
     (let* ((answer (torch-dojo-scenario-bridge-dispatch
                     '((id . 8) (command . "next") (args . nil))))
            (snap (alist-get 'snapshot answer)))
       (should (= (alist-get 'id answer) 8))
       (should (eq (alist-get 'ok answer) t))
       (should (= (alist-get 'pending_index snap) 1))
       (should (equal (alist-get 'result (aref (alist-get 'steps snap) 0)) "ready")))
     (should (eq :false (alist-get 'ok (torch-dojo-scenario-bridge-dispatch
                                       '((id . 9) (command . "eval") (args . nil)))))))))

(ert-deftest dojo-bridge-restart-preserves-source-and-clears-history ()
  (dojo-bridge-with-run
   (cl-letf (((symbol-function 'torch-dojo-scenario-setup) (lambda () "ready")))
     (torch-dojo-scenario-bridge-set-source "cell-1" "print('keep')")
     (torch-dojo-scenario-bridge-dispatch '((id . 1) (command . "next")))
     (let* ((response (torch-dojo-scenario-bridge-dispatch
                       '((id . 2) (command . "restart"))))
            (snapshot (alist-get 'snapshot response)))
       (should (= (alist-get 'pending_index snapshot) 0))
       (should (equal (alist-get 'source (aref (alist-get 'steps snapshot) 2))
                      "print('keep')\n"))
       (should (eq :null (alist-get 'result (aref (alist-get 'steps snapshot) 0))))))))

(ert-deftest dojo-bridge-restart-reports-cleanup-failure ()
  (dojo-bridge-with-run
   (setq torch-dojo-scenario-client 'owned)
   (cl-letf (((symbol-function 'torch-dojo-scenario-transport-shutdown)
              (lambda (_client done) (funcall done nil "shutdown refused"))))
     (let* ((response (torch-dojo-scenario-bridge-dispatch
                       '((id . 3) (command . "restart"))))
            (snapshot (alist-get 'snapshot response)))
       (should (eq t (alist-get 'failed snapshot)))
       (should (file-exists-p (expand-file-name "cleanup-error" dir)))))))

(ert-deftest dojo-bridge-socket-rejects-bad-and-large-frames ()
  (dojo-bridge-with-run
   (let* ((server (torch-dojo-scenario-bridge-start))
          (output (generate-new-buffer " *dojo-bridge-test*"))
          (client (make-network-process
                   :name "dojo-bridge-test" :family 'local
                   :service (expand-file-name "bridge.sock" dir)
                   :buffer output)))
     (unwind-protect
         (progn
           (process-send-string client "{broken}\n")
           (process-send-string client (concat (make-string (1+ torch-dojo-scenario-bridge-limit) ?x) "\n"))
           (accept-process-output nil .3)
           (with-current-buffer output
             (should (string-match-p "Malformed request" (buffer-string)))
             (should (string-match-p "Request too large" (buffer-string))))
           (should (= torch-dojo-scenario-index 0)))
       (delete-process client)
       (delete-process server)
       (kill-buffer output)))))

(ert-deftest dojo-bridge-edit-by-id-keeps-completed-result-until-restart ()
  (dojo-bridge-with-run
   (setq torch-dojo-scenario-index 4
         torch-dojo-scenario-history '((3 . "previous output")))
   (let* ((answer (torch-dojo-scenario-bridge-dispatch
                   '((id . 12) (command . "set_source")
                     (args . ((cell_id . "cell-2")
                              (source . "print('こんにちは')\nprint(2)"))))))
          (steps (alist-get 'steps (alist-get 'snapshot answer))))
     (should (eq t (alist-get 'ok answer)))
     (should (equal (alist-get 'source (aref steps 3))
                    "print('こんにちは')\nprint(2)\n"))
     (should (equal (alist-get 'result (aref steps 3)) "previous output"))
     (should (eq t (alist-get 'stale (aref steps 3))))
     (should (equal (alist-get 'source (aref steps 2))
                    (torch-dojo-scenario-org-source "cell-1"))))
   (should (eq :false (alist-get 'ok
                                (torch-dojo-scenario-bridge-dispatch
                                 '((id . 13) (command . "set_source")
                                   (args . ((cell_id . "cell-2")
                                            (source . "#+END_SRC\n"))))))))
   (should (eq :false (alist-get 'ok
                                (torch-dojo-scenario-bridge-dispatch
                                 '((id . 14) (command . "set_source")
                                   (args . ((cell_id . "missing")
                                            (source . "print(1)"))))))))
   (should (eq :false (alist-get 'ok
                                (torch-dojo-scenario-bridge-dispatch
                                 `((id . 15) (command . "set_source")
                                   (args . ((cell_id . "cell-2")
                                            (source . ,(make-string (1+ torch-dojo-scenario-bridge-limit) ?x)))))))))))

(ert-deftest dojo-bridge-cell-error-has-structured-traceback ()
  (dojo-bridge-with-run
   (cl-letf (((symbol-function 'torch-dojo-scenario-setup) (lambda () "ready"))
             ((symbol-function 'torch-dojo-scenario-transport-connect)
              (lambda (yes _no) (funcall yes 'client "kernel")))
             ((symbol-function 'torch-dojo-scenario-transport-execute)
              (lambda (_client _code _yes no)
                (funcall no 'error "Traceback (most recent call last):\nValueError: broken"))))
     (dotimes (_ 3) (torch-dojo-scenario-next))
     (let* ((step (aref (alist-get 'steps (torch-dojo-scenario-snapshot)) 2))
            (output (alist-get 'output step)))
       (should (equal (alist-get 'state step) "failed"))
       (should (string-match-p "ValueError: broken" (alist-get 'error output)))
       (should (equal (alist-get 'stdout output) ""))))))

;;; scenario_bridge_test.el ends here
