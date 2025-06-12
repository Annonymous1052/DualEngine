package com.example.dualengine

import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.os.Build
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat

// Activity but prevents direct execution and can only be used
// in inherited activities (implementations) by defining as 'abstract class'
abstract class BaseActivity: AppCompatActivity() {
    // Forces implementation in inherited activities
    abstract fun permissionGranted(requestCode :Int)
    abstract fun permissionDenied(requestCode :Int)

    // Method called directly when requesting permissions from child activities
    // Parameters: permission array to request, request code
    fun requirePermissions(permissions: Array<String>, requestCode: Int){
        // Android version check
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.M){ // Below 6.0
            permissionGranted(requestCode)
        }
        else{ // Version that needs permission check
            // Check if all permissions are granted
            //val isAllPermissionsGranted = permissions.all{
            //    checkSelfPermission(it) == PackageManager.+PERMISSION_GRANTED
            //}
            // Branch according to permission approval status
            //if (isAllPermissionsGranted){ // If all are approved,
            //    permissionGranted(requestCode)
            //}
            //else{ // If there are unapproved permissions,
            //    ActivityCompat.requestPermissions(this, permissions, requestCode)
            //}

        }
    }

    @SuppressLint("MissingSuperCall")
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        if (grantResults.all{ it == PackageManager.PERMISSION_GRANTED}){
            permissionGranted(requestCode)
        }
        else{
            permissionDenied(requestCode)
        }
    }

}