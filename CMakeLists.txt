add_executable(crop_images crop_images.cc)
target_link_libraries(crop_images -pthread ${BOOST_FILESYSTEM_LIB} ${BOOST_SYSTEM_LIB} ${BOOST_THREAD_LIB} ${BOOST_SYSTEM_LIB} ${BOOST_THREAD_LIB} ${OPENCV_CORE_LIB} ${OPENCV_HIGHGUI_LIB} ${OPENCV_IMGPROC_LIB} ${OPENCV_FEATURES2D_LIB} ${OPENCV_NONFREE_LIB})

add_library(directory directory.cc)
target_link_libraries(directory ${BOOST_SYSTEM_LIB} ${BOOST_FILESYSTEM_LIB})

add_executable(extract_feature extract_feature.cc)
target_link_libraries(extract_feature directory -pthread ${BOOST_FILESYSTEM_LIB} ${BOOST_SYSTEM_LIB} ${BOOST_THREAD_LIB} ${BOOST_SYSTEM_LIB} ${BOOST_THREAD_LIB} ${OPENCV_CORE_LIB} ${OPENCV_HIGHGUI_LIB} ${OPENCV_IMGPROC_LIB} ${OPENCV_FEATURES2D_LIB} ${OPENCV_NONFREE_LIB} ${OPENCV_OBJDETECT_LIB} ${OPENCV_CALIB3D_LIB})

add_executable(image_test image_test.cc)
target_link_libraries(image_test directory -pthread ${BOOST_FILESYSTEM_LIB} ${BOOST_SYSTEM_LIB} ${BOOST_THREAD_LIB} ${BOOST_SYSTEM_LIB} ${BOOST_THREAD_LIB} ${OPENCV_CORE_LIB} ${OPENCV_HIGHGUI_LIB} ${OPENCV_IMGPROC_LIB} ${OPENCV_FEATURES2D_LIB} ${OPENCV_NONFREE_LIB} ${OPENCV_OBJDETECT_LIB} ${OPENCV_CALIB3D_LIB})
