
def image_pick():
    import os
    import random
    import matplotlib.pyplot as plt
    import numpy as np

    dataset_dir ='C:/Users/jonghyun/Desktop/use_mini/train'          # 64개 class 사진 경로
    mini_class = os.listdir(dataset_dir)                             # class 64개 이름    
    class_five=random.sample(mini_class,5)                           # 64개 class 에서 5개 랜덤으로 폴더 뽑기

    all_support=[]                                                  #모든 support data 넣어줄 list  /  모든 train support_data들 25개 [...([행렬],'class')...]  
    all_support_name=[]   
    all_six=[]                                                      #                                  모든 image들 3000개 [...([행렬],'class')...] 
    all_six_name=[]

    for b in range(5):                                               #5개 클래스 랜덤으로 한개 뽑아줌
        class_dir = dataset_dir + '/' + class_five[b]                #class들중 뽑아진 한개의 폴더 경로
        one_class_sample=os.listdir(class_dir)                       #한 class 안의 모든 image들
        sample_five = random.sample(one_class_sample,5)              #한 클래스 안에서 image 5개만 random 뽑기
    

        train_support=[]                                             # 전체 support data 25개 하나씩 추가해줄 list

        for a in  range(5):                                          # 5개의 뽑힌 데이터에 대해서
            image_dir= class_dir + '/' + sample_five[a]              # 5개 image의 경로
            show=plt.imread(image_dir)                               # 그 image들을 행렬로 저장        
            data = (show, class_five[b])                             # 데이터를 (이미지행렬, class)로 저장
            train_support.append(data)                               # suppport data 에 하나씩 data를 추가
            all_support_name.append(image_dir)                                               # 25개 경로들만 따로 저장
        all_support.extend(train_support)                            # 만들어둔 all_support 안에 train_support 들을 넣어주기
    #all_support 끝    
        pre_six=[]                                                   # 3000개 image 하나씩 추가해줄 리list

        for i in range(600):                                         # 한 클래스 600개 전체에서
            all_img_dir = class_dir + '/' + one_class_sample[i]            # 모든 이미지의 경로
            all_show=plt.imread(all_img_dir)                         # 모든 이미지 행렬로 저장
            all_data = (all_show, class_five[b])                     # 데이터를 (이미지행렬, class)로 저장
            pre_six.append(all_data)                                 # class 하나에 대해 600개 다 저장
            all_six_name.append(all_img_dir) 
        all_six.extend(pre_six)                                      # 모든 class 에 대해 3000개 저장
    
    ###############  3000개에서 support data로 사용된 25개 빼기  #################

    #all_six_name.remove(all_support_name)                      #3000개 image 중 25개 제외
                                                                #이제 all_six_name 은 2750개 
    for x in all_support_name:
        all_six_name.remove(x)    

    train_query = random.sample(all_six_name,6)
    real_query_class = []

    for t in range(6):
        ex = str(train_query[t])
        query_path=(ex[41:50])
        real_query_class.append(query_path)



    final_query=[]
    # ( , real query class)
    for v in  range(6):                                          # 5개의 뽑힌 데이터에 대해서
            show=plt.imread(train_query[v])                      # 그 image들을 행렬로 저장        
            query = (show, real_query_class[v])                  # 데이터를 (이미지행렬, class)로 저장
            final_query.append(query)                            # suppport data 에 하나씩 data를 추가
