package com.example.objectdetector

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.MediaStore
import androidx.core.app.ActivityCompat
import com.example.objectdetector.ml.MobilenetV110224Quant
import kotlinx.android.synthetic.main.activity_main.*
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class MainActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        if(ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA)!=PackageManager.PERMISSION_GRANTED){
            button.isEnabled=false
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA),111)
        }
        else{
            button.isEnabled=true
            button.setOnClickListener{
                var i= Intent(MediaStore.ACTION_IMAGE_CAPTURE)
                startActivityForResult(i,101)
            }
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if(requestCode==101){
            var fileName="labels.txt"
            val inputString=application.assets.open(fileName).bufferedReader().use { it.readText() }
            var label_list=inputString.split("\n")
            var pic=data?.getParcelableExtra<Bitmap>("data")
            imageView2.setImageBitmap(pic)
            var resized_pic= pic?.let { Bitmap.createScaledBitmap(it,224,224,true) }
            val model = MobilenetV110224Quant.newInstance(this)

// Creates inputs for reference.
            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.UINT8)

            var tbuffer=TensorImage.fromBitmap(resized_pic)
            var byteBuffer=tbuffer.buffer
            inputFeature0.loadBuffer(byteBuffer)

// Runs model inference and gets result.
            val outputs = model.process(inputFeature0)
            val outputFeature0 = outputs.outputFeature0AsTensorBuffer
//            textView3.text=label_list[outputFeature0.floatArray.maxOrNull()?.toInt()!!].toString()
            var max=getMax(outputFeature0.floatArray)
            textView3.text=label_list[max].toString()
// Releases model resources if no longer used.
            model.close()

        }
    }
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if(requestCode==111&&grantResults[0]==PackageManager.PERMISSION_GRANTED){
            button.isEnabled=true
        }
    }
    fun getMax(arr:FloatArray):Int{
        var ind=0
        var min=0.0f
        for(i in 0..1000){
            if(arr[i]>min){
                ind=i
                min=arr[i]
            }
        }
        return ind
    }
}