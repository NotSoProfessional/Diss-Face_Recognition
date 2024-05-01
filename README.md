# Final Year Project: Optimising and Evaluating Face Recognition Methods on Android.

This project looks at various face recognition methods, and various ways of implementing them on Android to find a balance of speed and CPU utilization. The goal was to determine whether on device processing for face grouping is a viable option as currently Android devices using Google Photos perform this processing in the cloud, which is slow and is open to privacy and security concerns.

The implementation of the MobileFaceNet model was built upon Esteban Uri's Android application who originally converted the model from TensorFlow to TensorFlow Lite for use on mobile. Modifications to his code was mostly from updating from TFLite 2.3.0 to TFLite 2.9.0.

The application is known to work as far back as Android 8.0 which at the time (2022/2023) covered the large majority of Android devices.

Other forms of face recognition were done using the OpenCV library. Included are both x86_64 and Arm versions for greater compatibility and easier development.

Face Detection was implemented using Google's ML Kit with code being taken from their CodeLabs examples.

Image pre-processing was implemented twice, once using exclusively Java and Android's Graphics Library and again using OpenCV to compare performance.

While some config options can be changed using the UI, many in this version can only be changed in the code and recompiling, apologies.

Benchmarks results, and stats can only be seen within Logcat.

Quick takeways for the most accurate and quickest face detection recognition use the fast preset for Google's Face Detection, if you only require basic image pre-processing such as cropping and face alignment then use the Android Graphics Library. Face recognition with MFN is quickest and most power efficient using the XNNPack delegate on a single thread which is fast enough for real-time (30fps) recognition on images containing one face. Please keep in mind an NPU was not tested due to the lack of hardware resources and could be faster, however I doubt this as I did find any improvement from running the model on the GPU and believe the MFN model would have to be modified to make efficient use of these resources.

As the code did not contribute to the mark it is a mess and almost all of it needs refactoring and commenting, sorry in advance.
