;;; init.el -*- lexical-binding: t; -*-
(doom! :emacs dired electric undo vc
       :tools eval lookup
       :lang emacs-lisp (org +jupyter) python sh
       :config (default +bindings))
