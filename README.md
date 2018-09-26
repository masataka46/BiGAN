# BiGAN    

# discription  
 Implementation of BiGAN using tensorflow.   If any bug, please send me e-mail.  You have to  download MNIST.npz from official site.  
 https://github.com/houssamzenati/Efficient-GAN-Anomaly-Detection/tree/master/data  
 
# official implementation  
official implementation is here
[houssamzenati/Efficient-GAN-Anomaly-Detection](https://github.com/houssamzenati/Efficient-GAN-Anomaly-Detection)

# literature  
 [BiGAN](https://arxiv.org/abs/1802.06222)  

# dependency  
I confirmed operation only with..   
1)python 3.6.3  
2)tensorflow 1.7.0  
3)numpy 1.14.2    
4)Pillow 4.3.0  

# cost function  
<img width="915" alt="bigan_cost" src="https://user-images.githubusercontent.com/15444879/46053707-ef527a00-c17e-11e8-8644-891cbeece859.png">

# socore function  
<img width="366" alt="bigan_score" src="https://user-images.githubusercontent.com/15444879/46053719-f5485b00-c17e-11e8-93c9-84fd58cecf57.png">

# result Image
I have learned this model with normal '5'  from training data, and validate normal '5' from training data, and anormal '7' from training data, and validation data.  
after 990 epochs, the original image and reconstructed image is bellow.  
![resultimage_18092602_990](https://user-images.githubusercontent.com/15444879/46053638-ac90a200-c17e-11e8-98fb-d68ca21cba0d.png)

left to right .... original '5' (normal data), reconstructed '5', original '7' (annormal data), reconstructed '7'.  

# email  
t.ohmasa@w-farmer.com  
