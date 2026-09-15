import tensorflow as tf
import sys
 
def printsave(*a):
    file = open('c:\\a.txt','a')
    print(*a)
    print(*a,file=file)
    file.close()
# OR 데이터 구축
x=[[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]]
y=[[-1],[1],[1],[1]]

# 가중치 초기화
w=tf.Variable(tf.random.uniform([2,1],-0.5,0.5))
opt=tf.keras.optimizers.SGD(learning_rate=0.1)

# 전방 계산(식 (4.3))
def forward():
    s=tf.add(tf.matmul(x,w),b)   # x*w + b
    o=tf.tanh(s)
    return o

b=tf.Variable(tf.zeros([1]))

# 옵티마이저
# 손실 함수 정의
def loss():
    o=forward()
    #print(o)
    return tf.reduce_mean((y-o)**2)
output = "result"
# 500세대까지 학습(100세대마다 학습 정보 출력)
for i in range(500):
    opt.minimize(loss, var_list=[w,b])
    print("%f__x1+%f__x2+%f=0" %(w[0]/2.,w[1]/2.,b/2.) )
    printsave("%f__x1+%f__x2+%f=0" %(w[0]/2.,w[1]/2.,b/2.))
    if(i%100==0): print('loss at epoch',i,'=',loss().numpy()) 
                        
with open("output.txt", "w", encoding="utf-8") as file:
    file.write(output)                     

# 학습된 퍼셉트론으로 OR 데이터를 예측

o=forward()
print("%f__x1+%f__x2+%f=0" %(w[0]/2.,w[1]/2.,b/2.) )
#print(b) 
#print(o)