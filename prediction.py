from scipy import spatial
import numpy as np
import math
import torch    
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import matplotlib.pyplot as plt

from all_added_five import five_s_one_q

list = five_s_one_q()

pbar_one = torch.tensor(list[0])
pbar_two = torch.tensor(list[1])
pbar_three = torch.tensor(list[2])
pbar_four = torch.tensor(list[3])
pbar_five = torch.tensor(list[4])

qbar_one = torch.tensor(list[5])
qbar_two = torch.tensor(list[6])
qbar_three = torch.tensor(list[7])
qbar_four = torch.tensor(list[8])
qbar_five = torch.tensor(list[9])

# 64개의 trainset 안에서 5개를 뽑고 그 안에서 5개 뽑은것 끼리 한 class 의 support feature 만듬
# 5개 class 안에 남은 데이터들중 6개 랜덤으로 뽑은 쿼리데이터중 1개가 class들과 비교된 query feaure
gap=nn.AvgPool2d(6)

fir_loss_one=np.empty((1),float)
fir_loss_two=np.empty((1),float)
fir_loss_three=np.empty((1),float)
fir_loss_four=np.empty((1),float)
fir_loss_five=np.empty((1),float)

for i in range (6):
    for j in range (6):
        up_one = 1 - spatial.distance.cosine( qbar_one[:,i,j],gap(pbar_one) )           #1개짜리 tensor
        down =(
              1 - spatial.distance.cosine( qbar_one[:,i,j],gap(pbar_one) ) 
            + 1 - spatial.distance.cosine( qbar_two[:,i,j],gap(pbar_two) ) 
            + 1 - spatial.distance.cosine( qbar_three[:,i,j],gap(pbar_three) ) 
            + 1 - spatial.distance.cosine( qbar_four[:,i,j],gap(pbar_four) ) 
            + 1 - spatial.distance.cosine( qbar_five[:,i,j],gap(pbar_five) )
            )

        up_two = 1 - spatial.distance.cosine( qbar_two[:,i,j],gap(pbar_two) )     
        down =(
              1 - spatial.distance.cosine( qbar_one[:,i,j],gap(pbar_one) ) 
            + 1 - spatial.distance.cosine( qbar_two[:,i,j],gap(pbar_two) ) 
            + 1 - spatial.distance.cosine( qbar_three[:,i,j],gap(pbar_three) ) 
            + 1 - spatial.distance.cosine( qbar_four[:,i,j],gap(pbar_four) ) 
            + 1 - spatial.distance.cosine( qbar_five[:,i,j],gap(pbar_five) )
            )

        up_three = 1 - spatial.distance.cosine( qbar_three[:,i,j],gap(pbar_three) )     
        down =(
              1 - spatial.distance.cosine( qbar_one[:,i,j],gap(pbar_one) ) 
            + 1 - spatial.distance.cosine( qbar_two[:,i,j],gap(pbar_two) ) 
            + 1 - spatial.distance.cosine( qbar_three[:,i,j],gap(pbar_three) ) 
            + 1 - spatial.distance.cosine( qbar_four[:,i,j],gap(pbar_four) ) 
            + 1 - spatial.distance.cosine( qbar_five[:,i,j],gap(pbar_five) )
            )

        up_four = 1 - spatial.distance.cosine( qbar_four[:,i,j],gap(pbar_four) )     
        down =(
              1 - spatial.distance.cosine( qbar_one[:,i,j],gap(pbar_one) ) 
            + 1 - spatial.distance.cosine( qbar_two[:,i,j],gap(pbar_two) ) 
            + 1 - spatial.distance.cosine( qbar_three[:,i,j],gap(pbar_three) ) 
            + 1 - spatial.distance.cosine( qbar_four[:,i,j],gap(pbar_four) ) 
            + 1 - spatial.distance.cosine( qbar_five[:,i,j],gap(pbar_five) )
            )

        up_five = 1 - spatial.distance.cosine( qbar_five[:,i,j],gap(pbar_five) )     
        down =(
              1 - spatial.distance.cosine( qbar_one[:,i,j],gap(pbar_one) ) 
            + 1 - spatial.distance.cosine( qbar_two[:,i,j],gap(pbar_two) ) 
            + 1 - spatial.distance.cosine( qbar_three[:,i,j],gap(pbar_three) ) 
            + 1 - spatial.distance.cosine( qbar_four[:,i,j],gap(pbar_four) ) 
            + 1 - spatial.distance.cosine( qbar_five[:,i,j],gap(pbar_five) )
            )

        percent_one = np.log(up_one/down)
        percent_two = np.log(up_two/down)
        percent_three = np.log(up_three/down)
        percent_four = np.log(up_four/down)
        percent_five = np.log(up_five/down)

        fir_loss_one=fir_loss_one- percent_one
        fir_loss_two=fir_loss_two- percent_two
        fir_loss_three=fir_loss_three- percent_three
        fir_loss_four=fir_loss_four- percent_four
        fir_loss_five=fir_loss_five- percent_five                                        # first_loss          

fir_loss_one=np.squeeze(fir_loss_one)                                                                
fir_loss_two=np.squeeze(fir_loss_two)
fir_loss_three=np.squeeze(fir_loss_three)
fir_loss_four=np.squeeze(fir_loss_four)
fir_loss_five=np.squeeze(fir_loss_five)                                                  #나중을 위해 차수를 맞춰줌


#print(fir_loss_one)
#print(fir_loss_two)
#print(fir_loss_three)
#print(fir_loss_four)
#print(fir_loss_five)
##################################################       first_loss
#5개중 하나 쓰임 실제 클래스가 뭐냐에 따라.
query_class=list[10][0]                                 # return 의 11번째 값인 실제 쿼리 클래스
support_class=list[11]                                  # support class 5개

print("query class : " + query_class)
print("candidate support classes : ")
print(support_class)                                     
#한번 백프롭 하기 전에 6개 쿼리 실험해봐야함
# 다 더하는게 아니고 로그 씌운후 더하기

model_g=torch.nn.Sequential(nn.Linear(512, 5))
probability=[]
for i in range (6):
    for j in range (6):
        before=model_g(qbar_four[:,i,j])                #[36,1,1] => 5 
        before=before.detach().numpy()
        probability.append(before)
        
probability=torch.tensor(probability)                   # 길이36
                                                                   
z=-F.log_softmax(probability, dim=1)                    # softmax 랑 log 해줌 
z=z.numpy()                                             # 크기(36,5)
                                          
sec_loss_one = np.sum(z[:,0])                           
sec_loss_two = np.sum(z[:,1])
sec_loss_three = np.sum(z[:,2])
sec_loss_four = np.sum(z[:,3])
sec_loss_five = np.sum(z[:,4])

#print(sec_loss_one)
#print(sec_loss_two)
#print(sec_loss_three)
#print(sec_loss_four)
#print(sec_loss_five)                                                         # Secondd_loss

#print((fir_loss_one*0.5) + sec_loss_one)
#print((fir_loss_two*0.5) + sec_loss_two)
#print((fir_loss_three*0.5) + sec_loss_three)
#print((fir_loss_four*0.5) + sec_loss_four)
#print((fir_loss_five*0.5) + sec_loss_five)                                   #최종 loss 값

loss_list=[(fir_loss_one*0.5) + sec_loss_one , (fir_loss_two*0.5) + sec_loss_two , (fir_loss_three*0.5) + sec_loss_three , (fir_loss_four*0.5) + sec_loss_four , (fir_loss_five*0.5) + sec_loss_five]

class_dict = {(fir_loss_one*0.5) + sec_loss_one : support_class[0],          #loss랑 class이름이랑 붙여줌
        (fir_loss_two*0.5) + sec_loss_two : support_class[1],
        (fir_loss_three*0.5) + sec_loss_three : support_class[2],
        (fir_loss_four*0.5) + sec_loss_four : support_class[3],
        (fir_loss_five*0.5) + sec_loss_five : support_class[4],
        }

print("class prediction :" + class_dict[np.min(loss_list)])                                         # 제일 loss 가 작은 class의 이름

if class_dict[np.min(loss_list)] == query_class:
    print("correct prediction")
else:
    print("wrong prediction")
    