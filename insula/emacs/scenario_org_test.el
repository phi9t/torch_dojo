;;; scenario_org_test.el -*- lexical-binding: t; -*-
(require 'ert)
(load-file "/workspace/torch_dojo/insula/emacs/scenario_org.el")

(ert-deftest dojo-org-discovers-unnamed-nested-cell-and-keeps-ids-after-edit ()
  (with-temp-buffer
    (org-mode)
    (insert "* Lesson\nSome context.\n#+NAME: first\n#+BEGIN_SRC python\nprint(1)\n#+END_SRC\n** More\n#+BEGIN_SRC bash\necho ignored\n#+END_SRC\n#+BEGIN_SRC python\nprint(2)\n#+END_SRC\n")
    (let ((cells (torch-dojo-scenario-org-discover)))
      (should (= (length cells) 2))
      (should (equal (alist-get 'id (aref cells 1)) "cell-2"))
      (should (string-match-p "More" (alist-get 'heading (aref cells 1))))
      (should (= (torch-dojo-scenario-org-skipped-count) 1))
      (torch-dojo-scenario-org-set-source "cell-2" "print(3)")
      (should (equal (torch-dojo-scenario-org-source "cell-2") "print(3)\n"))
      (should (equal (alist-get 'id (aref (torch-dojo-scenario-org-discover) 1)) "cell-2")))))

(ert-deftest dojo-org-rejects-empty-and-unsafe-headers ()
  (with-temp-buffer
    (org-mode)
    (insert "* Empty\n#+BEGIN_SRC bash\necho hi\n#+END_SRC\n")
    (should-error (torch-dojo-scenario-org-discover)))
  (with-temp-buffer
    (org-mode)
    (insert "* Unsafe\n#+BEGIN_SRC python :var x=1\nprint(x)\n#+END_SRC\n")
    (should-error (torch-dojo-scenario-org-discover) :type 'error)))

(ert-deftest dojo-org-rejects-boundaries-and-external-structure-change ()
  (with-temp-buffer
    (org-mode)
    (insert "* Cell\n#+BEGIN_SRC python\nprint(1)\n#+END_SRC\n")
    (torch-dojo-scenario-org-discover)
    (should-error (torch-dojo-scenario-org-set-source "cell-1" "#+END_SRC\n"))
    (goto-char (point-max))
    (insert "#+BEGIN_SRC python\nprint(2)\n#+END_SRC\n")
    (should-error (torch-dojo-scenario-org-source "cell-1"))))

(ert-deftest dojo-org-rejects-external-heading-change-before-edit ()
  (with-temp-buffer
    (org-mode)
    (insert "* First\n#+BEGIN_SRC python\nprint(1)\n#+END_SRC\n")
    (torch-dojo-scenario-org-discover)
    (goto-char (point-min))
    (delete-region (line-beginning-position) (line-end-position))
    (insert "* Different")
    (should-error (torch-dojo-scenario-org-set-source "cell-1" "print(2)"))))

(ert-deftest dojo-org-clears-only-python-results ()
  (with-temp-buffer
    (org-mode)
    (insert "* Mix\n#+BEGIN_SRC bash\necho old\n#+END_SRC\n#+RESULTS:\n: keep bash\n#+BEGIN_SRC python\nprint(1)\n#+END_SRC\n#+RESULTS:\n: old python\n")
    (torch-dojo-scenario-org-discover)
    (torch-dojo-scenario-org-clear-results)
    (should (string-match-p "keep bash" (buffer-string)))
    (should-not (string-match-p "old python" (buffer-string)))))

(ert-deftest dojo-org-discovers-context-for-each-cell-in-document-order ()
  (with-temp-buffer
    (org-mode)
    (insert "* Lesson\nRead this first.\n\n#+NAME: alpha\n#+BEGIN_SRC python :results output\nprint(1)\n#+END_SRC\n\nA second idea with [[https://example.org][a link]].\n#+BEGIN_SRC python\nprint(2)\n#+END_SRC\n** Deep\nNested explanation.\n| a | b |\n#+BEGIN_SRC sh\necho skipped\n#+END_SRC\n#+BEGIN_SRC python\nprint(3)\n#+END_SRC\n")
    (let ((cells (torch-dojo-scenario-org-discover)))
      (should (equal (mapcar (lambda (cell) (alist-get 'id cell)) (append cells nil))
                     '("cell-1" "cell-2" "cell-3")))
      (should (equal (alist-get 'name (aref cells 0)) "alpha"))
      (should (string-match-p "Read this first" (alist-get 'text (aref cells 0))))
      (should (string-match-p "A second idea" (alist-get 'text (aref cells 1))))
      (should (string-match-p "Lesson / Deep" (alist-get 'heading (aref cells 2))))
      (should (string-match-p "Nested explanation" (alist-get 'text (aref cells 2))))
      (should (= (torch-dojo-scenario-org-skipped-count) 1)))))

(ert-deftest dojo-org-replaces-table-list-drawer-and-colon-results-without-prose-loss ()
  (dolist (result '("| old | table |\n| 1 | 2 |\n"
                    "- old list\n- second item\n"
                    ":RESULTS:\nold drawer\n:END:\n"
                    ": old colon\n: second line\n"))
    (with-temp-buffer
      (org-mode)
      (insert (concat "* Lesson\n#+BEGIN_SRC python\nprint(1)\n#+END_SRC\n\n"
                      "#+RESULTS:\n" result "\nKeep this prose.\n"
                      "#+BEGIN_SRC bash\necho other\n#+END_SRC\n#+RESULTS:\n: keep bash\n"))
      (torch-dojo-scenario-org-discover)
      (torch-dojo-scenario-org-write-result "cell-1" "fresh")
      (should (= (how-many "^#\\+RESULTS:" (point-min) (point-max)) 2))
      (should (string-match-p (regexp-quote "#+RESULTS:\n: fresh") (buffer-string)))
      (should (string-match-p "Keep this prose" (buffer-string)))
      (should (string-match-p "keep bash" (buffer-string)))
      (should-not (string-match-p "old table\\|old list\\|old drawer\\|old colon" (buffer-string)))
      (torch-dojo-scenario-org-clear-results)
      (should (= (how-many "^#\\+RESULTS:" (point-min) (point-max)) 1))
      (should (string-match-p "Keep this prose" (buffer-string)))
      (should (string-match-p "keep bash" (buffer-string))))))
