package com.example.dualengine

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import com.example.dualengine.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {

    private val binding by lazy {ActivityMainBinding.inflate(layoutInflater)}

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(binding.root)

        binding.btnRunApp.setOnClickListener {
            val intent = Intent(this, PhotoActivity::class.java)
            startActivity(intent)
        }
    }
}