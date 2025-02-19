# libyt jupyter kernel config file
if (JUPYTER_KERNEL)
  set(LIBYT_KERNELSPEC_RELATIVE_PATH_TO_ROOT "share/jupyter/kernels/libyt_kernel")
  configure_file(
    "${PROJECT_SOURCE_DIR}/${LIBYT_KERNELSPEC_RELATIVE_PATH_TO_ROOT}/kernel.json"
    "${PROJECT_BINARY_DIR}/${LIBYT_KERNELSPEC_RELATIVE_PATH_TO_ROOT}/kernel.json"
    COPYONLY
  )
endif ()

# INSTALLATION RULE #
if (JUPYTER_KERNEL)
  include(GNUInstallDirs)
  install(
    FILES "${PROJECT_BINARY_DIR}/${LIBYT_KERNELSPEC_RELATIVE_PATH_TO_ROOT}/kernel.json"
    DESTINATION "${LIBYT_KERNELSPEC_RELATIVE_PATH_TO_ROOT}"
  )
endif ()
