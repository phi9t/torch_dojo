;;; verify_scenario.el --- Batch check of the interactive scenario -*- lexical-binding: t; -*-
(require 'scenario)

(torch-dojo-scenario-open)
(let ((owned-id nil)
      (step-count (length torch-dojo-scenario-steps))
      (cell-count (- (length torch-dojo-scenario-steps) 4)))
  (condition-case err
      (progn
        (dotimes (step step-count)
          (torch-dojo-scenario-next)
          (unless (and (= torch-dojo-scenario-index (1+ step))
                       (not torch-dojo-scenario-failed))
            (error "Step %d failed: %s" (1+ step)
                   torch-dojo-scenario-last-error))
          (when (= step 1) (setq owned-id torch-dojo-scenario-kernel-id))
          (when (and (> step 1) (< step (1- step-count)))
            (unless (equal owned-id torch-dojo-scenario-kernel-id)
              (error "Kernel ID changed between cells")))
          (princ (format "checkpoint=%d %s result=%s\n"
                         (1+ step) (torch-dojo-scenario-step-name step)
                         (cdr (assq step torch-dojo-scenario-history)))))
        (unless (and (string-match-p
                      "executable=/usr/bin/python"
                      (cdr (assq 2 torch-dojo-scenario-history)))
                     (string-match-p
                      "cwd=/workspace/torch_dojo"
                      (cdr (assq 2 torch-dojo-scenario-history)))
                     (string-match-p
                      "device=cuda:0 sum=10.0"
                      (cdr (assq 3 torch-dojo-scenario-history)))
                     (string-match-p
                      "visible from jupyter"
                      (cdr (assq 4 torch-dojo-scenario-history)))
                     (string-match-p (format "completed-cells=%d" cell-count)
                                     (cdr (assq (- step-count 2) torch-dojo-scenario-history)))
                     (torch-dojo-scenario-kernel-gone-p owned-id))
          (error "Scenario evidence missing"))
        (princ "scenario verified\n")
        (torch-dojo-scenario-quit))
    (error
     (princ (format "scenario failed: %s\n" (error-message-string err)))
     (setq torch-dojo-scenario-failed t)
     (torch-dojo-scenario-quit))))
