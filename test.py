if __name__=="__main__":

    import subprocess
    subprocess.run("mvNCCompile -s 12 -o test_3  optimized.pb  -in=input -on=MobilenetV1/Predictions/Reshape_1".split())