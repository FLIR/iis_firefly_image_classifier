. /opt/movidius/virtualenv-python/bin/activate

mvNCCompile -s 12 -o test_3  optimized.pb  -in=input -on=MobilenetV1/Predictions/Reshape_1
