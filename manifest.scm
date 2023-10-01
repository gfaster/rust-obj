(use-modules (guix packages)
             (guix git-download)
             (guix download)
             (guix gexp)
             (guix build-system cmake)
             (guix build-system copy)
             ((guix licenses) #:prefix license:)
             (gnu packages vulkan)
             (gnu packages autotools)
             (gnu packages bison)
             (gnu packages cmake)
             (gnu packages gl)
             (gnu packages pcre)
             (gnu packages qt)
             (gnu packages python)
             (gnu packages xorg)
             (gnu packages pkg-config)
             (gnu packages swig)
             (gnu packages gcc)
             (gnu packages commencement)
            )

(define shaderc-obj 
  (package 
    (inherit shaderc)
    (native-search-paths 
      (list (search-path-specification 
              (variable "SHADERC_LIB_DIR")
              (separator #f)
              (files (list "lib")))
            (search-path-specification 
              (variable "LD_LIBRARY_PATH")
              (separator #f)
              (files (list "lib")))
            ))
    ))

(define renderdoc-swig
  (package (inherit swig)
    (name "renderdoc-swig")
    (version "7")
    (source 
      (origin
        (method url-fetch)
        (uri (string-append "https://github.com/baldurk/swig/archive/renderdoc-modified-" version ".tar.gz"))
        ;; (sha256 (base32 "0j66jq2z3c9i6qvykj49kg0ch71cnm6s6rxibb4jxi56ml9m0zlx"))))
        (sha256 (base32 "19sb4vik8kgy9pxjfam8qc8i52iraj2971p1hrzh850fvl0nibg7"))))
    (inputs (list autoconf automake pcre python-wrapper))
    (arguments 
      `(
        #:make-flags (list (string-append "prefix=" (assoc-ref %outputs "out")))
        #:phases
        (modify-phases %standard-phases (delete 'check))
        ))
    ))

(define renderdoc
  (package
    (name "renderdoc")
    (version "1.29")
    (source 
      (origin
        (method git-fetch)
        (uri (git-reference 
               (url "https://github.com/baldurk/renderdoc")
               (commit (string-append "v" version))))
        (file-name (git-file-name name version))
        (patches (list (local-file "/home/gfaster/projects/obj/renderdoc-no-vendor-swig.patch")))
        (sha256 (base32 "1sjsqj9w2hka61i8b8fy440q8ba3mf376jr5lzpdwp4vx814q9jn"))))
        ;; (sha256 (base32 "01693qb8gzrf6k4aj1841bqwgymhndsmqawjwzs2pr9dxnkghi5w"))))
    (build-system cmake-build-system)
    (arguments 
      `(
       #:configure-flags ,#~(list (string-append "-DVULKAN_LAYER_FOLDER=" 
                                                #$output "/etc/vulkan"))
       ;; #:configure-flags #~(list "-DCMAKE_BUILD_TYPE=Release" "-Bbuild")
       ;; #:configure-flags ,#~(list 
       ;;                       (string-append "-DRENDERDOC_SWIG_PACKAGE=file://" 
       ;;                                      #$(this-package-native-input "renderdoc-swig")))
       ;; #:configure-flags #~(list "-DCMAKE_BUILD_TYPE=Release")
       #:phases 
       (modify-phases %standard-phases 
        (add-after 'unpack 'use-guix-dep 
          (lambda* (#:key inputs #:allow-other-keys)
                   (let ((rdocswig (assoc-ref inputs "renderdoc-swig")))
                     ;; (raise rdocswig)
            ;; (symlink "bin/swig" rdocswig);
            (substitute* "qrenderdoc/CMakeLists.txt"
               (("DEPENDS custom_swig")
                (string-append "DEPENDS " rdocswig "/bin/swig")))
            (substitute* "qrenderdoc/CMakeLists.txt"
               (("\\$\\{CMAKE_BINARY_DIR\\}/bin/swig")
                (string-append rdocswig "/bin/swig")))
            )))
        (delete 'check))

                ;; (string-append "DEPENDS " #$(this-package-native-input "renderdoc-swig")))))))
      ))
    (inputs
      (list libx11 libxcb xcb-util-keysyms mesa pkg-config
            python-wrapper bison autoconf automake python-pyside-2
            pcre qtbase-5 qtsvg-5 qtx11extras renderdoc-swig)
      )
    (propagated-inputs 
      (list qtbase-5))
    (synopsis "Stand-alone graphics debugging tool")
    (description "RenderDoc is a stand-alone graphics debugging tool.")
    (home-page "https://renderdoc.org/")
    (license license:expat)
    ))

;; (packages->manifest (list shaderc-obj (make-libstdc++ gcc) renderdoc))
(packages->manifest (list shaderc-obj renderdoc libxcb mesa qtbase-5 coreutils))
