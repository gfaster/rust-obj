(use-modules (guix config)
             (guix packages)
             (gnu packages vulkan)
             (gnu packages gcc)
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

(packages->manifest (list shaderc-obj (make-libstdc++ gcc)))
