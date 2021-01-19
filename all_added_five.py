def five_s_one_q():
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
    for v in  range(6):                                          # 6개의 뽑힌 데이터에 대해서
            show=plt.imread(train_query[v])                      # 그 image들을 행렬로 저장        
            query = (show, real_query_class[v])                  # 데이터를 (이미지행렬, class)로 저장
            final_query.append(query)                            # query data 에 하나씩 data를 추가

    #print(final_query)
    #resnet_12

    import math
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def conv3x3(in_planes, out_planes, stride=1):
        """3x3 convolution with padding"""
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=1, bias=False)


    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, inplanes, planes, kernel=3, stride=1, downsample=None):
            super(BasicBlock, self).__init__()
            if kernel == 1:
                self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
            elif kernel == 3:
                self.conv1 = conv3x3(inplanes, planes)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                   padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            if kernel == 1:
                self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
            elif kernel == 3:
                self.conv3 = conv3x3(planes, planes)
            self.bn3 = nn.BatchNorm2d(planes)
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample
            self.stride = stride

        def forward(self, x):
            residual = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                residual = self.downsample(x)

            out += residual
            out = self.relu(out)

            return out


    class Bottleneck(nn.Module):
        expansion = 4

        def __init__(self, inplanes, planes, kernel=1, stride=1, downsample=None):
            super(Bottleneck, self).__init__()
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                   padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(planes * 4)
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample
            self.stride = stride

        def forward(self, x):
            residual = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                residual = self.downsample(x)

            out += residual
            out = self.relu(out)

            return out


    class ResNet(nn.Module):

        def __init__(self, block, layers, kernel=3):
            self.inplanes = 64
            self.kernel = kernel
            super(ResNet, self).__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)

            self.layer1 = self._make_layer(block, 64, layers[0], stride=2) 
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

            self.nFeat = 512 * block.expansion

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

        def _make_layer(self, block, planes, blocks, stride=1):
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )

            layers = []
            layers.append(block(self.inplanes, planes, self.kernel, stride, downsample))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, self.kernel))

            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            return x


    def resnet12():                                                         
        model = ResNet(BasicBlock, [1,1,1,1], kernel=3)
        return model

    image=final_query[0][0]                                  # 0-5 총 6개 해주어야함

    image = image.transpose(2,0,1)[np.newaxis]               #[1,3,84,84] 로 만들어줌
    image = torch.tensor(image)                              # tensor로 만들어줌
    image = image.type('torch.FloatTensor')                  # float들로 숫자 저장

    model=resnet12()                                         #resnet불러오기
    query_feature=model(image)                               #resnet에 집어넣기

    # 서폿5개 레스넷 더한거 나누기5    (첫번째클래스 서폿셋)                                              
    temp=np.zeros((1,512,6,6))                                   # 5개 다 더할 temp 만들기
    temp=torch.tensor(temp)
    temp=temp.type('torch.FloatTensor')

    for p in range(0,5):                                         # 5-10 , 10-15 , 15-20 , 20-25 해주어야함 
        one_support = all_support[p][0]
        one_support = one_support.transpose(2,0,1)[np.newaxis]   #[1,3,84,84] 로 만들어줌
        one_support = torch.tensor(one_support)                  # tensor로 만들어줌
        one_support = one_support.type('torch.FloatTensor') 
    
        model=resnet12()
        added_feature=model(one_support)
        temp=temp+added_feature                                  # 0-4 5개 support data feature 합
    


    support_feature=temp/5                                       # 5개 feature의 평균

    support_feature=support_feature.view([-1, 6, 6])             # torch.Size([512, 6, 6])
    query_feature=query_feature.view([-1, 6, 6])                 # torch.Size([512, 6, 6])

    support_1=support_feature[:,0,0]
    support_2=support_feature[:,1,0]
    support_3=support_feature[:,2,0]
    support_4=support_feature[:,3,0]
    support_5=support_feature[:,4,0]
    support_6=support_feature[:,5,0]

    support_7=support_feature[:,0,1]
    support_8=support_feature[:,1,1]
    support_9=support_feature[:,2,1]
    support_10=support_feature[:,3,1]
    support_11=support_feature[:,4,1]
    support_12=support_feature[:,5,1]

    support_13=support_feature[:,0,2]
    support_14=support_feature[:,1,2]
    support_15=support_feature[:,2,2]
    support_16=support_feature[:,3,2]
    support_17=support_feature[:,4,2]
    support_18=support_feature[:,5,2]

    support_19=support_feature[:,0,3]
    support_20=support_feature[:,1,3]
    support_21=support_feature[:,2,3]
    support_22=support_feature[:,3,3]
    support_23=support_feature[:,4,3]
    support_24=support_feature[:,5,3]

    support_25=support_feature[:,0,4]
    support_26=support_feature[:,1,4]
    support_27=support_feature[:,2,4]
    support_28=support_feature[:,3,4]
    support_29=support_feature[:,4,4]
    support_30=support_feature[:,5,4]

    support_31=support_feature[:,0,5]
    support_32=support_feature[:,1,5]
    support_33=support_feature[:,2,5]
    support_34=support_feature[:,3,5]
    support_35=support_feature[:,4,5]
    support_36=support_feature[:,5,5]

    #########################################################################
    query_1=query_feature[:,0,0]
    query_2=query_feature[:,1,0]
    query_3=query_feature[:,2,0]
    query_4=query_feature[:,3,0]
    query_5=query_feature[:,4,0]
    query_6=query_feature[:,5,0]

    query_7=query_feature[:,0,1]
    query_8=query_feature[:,1,1]
    query_9=query_feature[:,2,1]
    query_10=query_feature[:,3,1]
    query_11=query_feature[:,4,1]
    query_12=query_feature[:,5,1]

    query_13=query_feature[:,0,2]
    query_14=query_feature[:,1,2]
    query_15=query_feature[:,2,2]
    query_16=query_feature[:,3,2]
    query_17=query_feature[:,4,2]
    query_18=query_feature[:,5,2]

    query_19=query_feature[:,0,3]
    query_20=query_feature[:,1,3]
    query_21=query_feature[:,2,3]
    query_22=query_feature[:,3,3]
    query_23=query_feature[:,4,3]
    query_24=query_feature[:,5,3]

    query_25=query_feature[:,0,4]
    query_26=query_feature[:,1,4]
    query_27=query_feature[:,2,4]
    query_28=query_feature[:,3,4]
    query_29=query_feature[:,4,4]
    query_30=query_feature[:,5,4]

    query_31=query_feature[:,0,5]
    query_32=query_feature[:,1,5]
    query_33=query_feature[:,2,5]
    query_34=query_feature[:,3,5]
    query_35=query_feature[:,4,5]
    query_36=query_feature[:,5,5]

    ############################################################################
    support_set = [support_1,support_2,support_3,support_4,support_5,support_6,support_7,support_8, support_9,support_10,support_11,support_12,support_13,support_14,  support_15,support_16, support_17,support_18,support_19,support_20, support_21,support_22, support_23,support_24,support_25,support_26,support_27,support_28, support_29,support_30,support_31,support_32,support_33,support_34,support_35,support_36]
    ############################################################################
    query_set = [query_1,query_2,query_3,query_4,query_5,query_6,query_7,query_8,query_9,query_10,query_11,query_12,query_13,query_14,query_15,query_16,query_17,query_18,query_19,query_20,query_21,query_22,query_23,query_24,query_25,query_26,query_27,query_28,query_29,query_30,query_31,query_32,query_33,query_34,query_35,query_36]

    result_dict = {}
    k=[]
    for i, support_i in enumerate(support_set):
        support_i=support_i / torch.norm(support_i, p='fro')   #벡터를 크기로 나누어주기(normalize) [512]
        support_i=torch.unsqueeze(support_i, 1)                #[512,1]
        trans_support_i=support_i.t()                          #[1,512]
        for j, query_j in enumerate(query_set):

            query_j=query_j / torch.norm(query_j, p='fro')         #[512]
            query_j=torch.unsqueeze(query_j, 1)                    #[512,1]

            r_i_j=torch.matmul(trans_support_i,query_j)                #[1]              총 36x36 개 r 나오게 해야함
        
            # 원하는것 : r_32 하면 스칼라값 하나가 나오는거
        
            name = f"r_{i+1}_{j+1}"
            result_dict[name] = r_i_j
            r_i_j=r_i_j.tolist() 
            k.append(r_i_j)

    R=np.array(k)

    R_q=R.reshape(36,6,6)                 #R_q.shape => (36,6,6)

    pre_p=R.reshape(6,36,6)
    R_p=np.transpose(pre_p,(1,2,0))       #R_p.shape => (36,6,6)

    R_p=torch.tensor(R_p)                 # numpy array 를 pytorch tensor 로 바꿔줌
    R_q=torch.tensor(R_q)           

    gap=nn.AvgPool2d(6)
    m_p=gap(R_p)
    m_q=gap(R_q)

    m_p=m_p.squeeze()
    m_p=m_p.type('torch.FloatTensor')

    m_q=m_q.squeeze()
    m_q=m_q.type('torch.FloatTensor')

    model = torch.nn.Sequential(
        nn.Linear(36, 6),
        nn.ReLU(),
        nn.Linear(6, 36)
    )

    w_p=model(m_p)
    w_q=model(m_q)
    #################################################################6_make_w

    w_p = w_p.type('torch.DoubleTensor')
    w_q = w_q.type('torch.DoubleTensor')

    bunmo_p=[0]
    A_p=[]

    for i in range (6):
        for j in range (6):
            p=torch.matmul(R_p[:,i,j],w_p)
            p=p.detach().numpy()
            temp=np.exp(p/0.025)
            bunmo_p=bunmo_p+temp                         # 하나씩 더해진 분모

    for i in range (6):
        for j in range (6):
            p=torch.matmul(R_p[:,i,j],w_p)
            p=p.detach().numpy()
            bunja=np.exp(p/0.025)
            A_p.append(bunja/bunmo_p)

    #print(A_p)
    #print(bunmo_p)

    ###################################################################################
    bunmo_q=[0]
    A_q=[]

    for i in range (6):
        for j in range (6):
            q=torch.matmul(R_q[:,i,j],w_q)
            q=q.detach().numpy()
            temp=np.exp(q/0.025)
            bunmo_q=bunmo_q+temp                         # 하나씩 더해진 분모

    for i in range (6):
        for j in range (6):
            q=torch.matmul(R_q[:,i,j],w_q)
            q=q.detach().numpy()
            bunja=np.exp(q/0.025)
            A_q.append(bunja/bunmo_q)

    #print(A_q)
    #print(bunmo_q)

    ###############################################################7 Ap Aq 구하기

    for x in range(36):
        A_p[x]=A_p[x]+1
        A_q[x]=A_q[x]+1
    
    # support_feature , A_p Elementwisely weight
    support_feature=support_feature.detach().numpy()
    query_feature=query_feature.detach().numpy()

    for i in range(6):
        for j in range(6):
            support_feature[:,i,j]=np.multiply(support_feature[:,i,j],A_p[j+(6*i)])
        
    for i in range(6):
        for j in range(6):
            query_feature[:,i,j]=np.multiply(query_feature[:,i,j],A_q[j+(6*i)])

    #print(support_feature.shape)
    #print(query_feature.shape)

    pbar_one=support_feature
    qbar_one=query_feature

    ##############################################################8_make_Pbar_Qbar_one

    image=final_query[0][0]                                  # 0-5 총 6개 해주어야함

    image = image.transpose(2,0,1)[np.newaxis]               #[1,3,84,84] 로 만들어줌
    image = torch.tensor(image)                              # tensor로 만들어줌
    image = image.type('torch.FloatTensor')                  # float들로 숫자 저장

    model=resnet12()                                         #resnet불러오기
    query_feature=model(image)                               #resnet에 집어넣기

    # 서폿5개 레스넷 더한거 나누기5    (첫번째클래스 서폿셋)                                              
    temp=np.zeros((1,512,6,6))                                   # 5개 다 더할 temp 만들기
    temp=torch.tensor(temp)
    temp=temp.type('torch.FloatTensor')


    for p in range(5,10):                                         # 5-10 , 10-15 , 15-20 , 20-25 해주어야함 
        one_support = all_support[p][0]
        one_support = one_support.transpose(2,0,1)[np.newaxis]   #[1,3,84,84] 로 만들어줌
        one_support = torch.tensor(one_support)                  # tensor로 만들어줌
        one_support = one_support.type('torch.FloatTensor') 
    
        model=resnet12()
        added_feature=model(one_support)
        temp=temp+added_feature                                  # 0-4 5개 support data feature 합
    


    support_feature=temp/5                                       # 5개 feature의 평균
    support_feature=support_feature.view([-1, 6, 6])             # torch.Size([512, 6, 6])
    query_feature=query_feature.view([-1, 6, 6])                 # torch.Size([512, 6, 6])

    support_1=support_feature[:,0,0]
    support_2=support_feature[:,1,0]
    support_3=support_feature[:,2,0]
    support_4=support_feature[:,3,0]
    support_5=support_feature[:,4,0]
    support_6=support_feature[:,5,0]

    support_7=support_feature[:,0,1]
    support_8=support_feature[:,1,1]
    support_9=support_feature[:,2,1]
    support_10=support_feature[:,3,1]
    support_11=support_feature[:,4,1]
    support_12=support_feature[:,5,1]

    support_13=support_feature[:,0,2]
    support_14=support_feature[:,1,2]
    support_15=support_feature[:,2,2]
    support_16=support_feature[:,3,2]
    support_17=support_feature[:,4,2]
    support_18=support_feature[:,5,2]

    support_19=support_feature[:,0,3]
    support_20=support_feature[:,1,3]
    support_21=support_feature[:,2,3]
    support_22=support_feature[:,3,3]
    support_23=support_feature[:,4,3]
    support_24=support_feature[:,5,3]

    support_25=support_feature[:,0,4]
    support_26=support_feature[:,1,4]
    support_27=support_feature[:,2,4]
    support_28=support_feature[:,3,4]
    support_29=support_feature[:,4,4]
    support_30=support_feature[:,5,4]

    support_31=support_feature[:,0,5]
    support_32=support_feature[:,1,5]
    support_33=support_feature[:,2,5]
    support_34=support_feature[:,3,5]
    support_35=support_feature[:,4,5]
    support_36=support_feature[:,5,5]

    #########################################################################
    query_1=query_feature[:,0,0]
    query_2=query_feature[:,1,0]
    query_3=query_feature[:,2,0]
    query_4=query_feature[:,3,0]
    query_5=query_feature[:,4,0]
    query_6=query_feature[:,5,0]

    query_7=query_feature[:,0,1]
    query_8=query_feature[:,1,1]
    query_9=query_feature[:,2,1]
    query_10=query_feature[:,3,1]
    query_11=query_feature[:,4,1]
    query_12=query_feature[:,5,1]

    query_13=query_feature[:,0,2]
    query_14=query_feature[:,1,2]
    query_15=query_feature[:,2,2]
    query_16=query_feature[:,3,2]
    query_17=query_feature[:,4,2]
    query_18=query_feature[:,5,2]

    query_19=query_feature[:,0,3]
    query_20=query_feature[:,1,3]
    query_21=query_feature[:,2,3]
    query_22=query_feature[:,3,3]
    query_23=query_feature[:,4,3]
    query_24=query_feature[:,5,3]

    query_25=query_feature[:,0,4]
    query_26=query_feature[:,1,4]
    query_27=query_feature[:,2,4]
    query_28=query_feature[:,3,4]
    query_29=query_feature[:,4,4]
    query_30=query_feature[:,5,4]

    query_31=query_feature[:,0,5]
    query_32=query_feature[:,1,5]
    query_33=query_feature[:,2,5]
    query_34=query_feature[:,3,5]
    query_35=query_feature[:,4,5]
    query_36=query_feature[:,5,5]

    ############################################################################
    support_set = [support_1,support_2,support_3,support_4,support_5,support_6,support_7,support_8, support_9,support_10,support_11,support_12,support_13,support_14,  support_15,support_16, support_17,support_18,support_19,support_20, support_21,support_22, support_23,support_24,support_25,support_26,support_27,support_28, support_29,support_30,support_31,support_32,support_33,support_34,support_35,support_36]
    ############################################################################
    query_set = [query_1,query_2,query_3,query_4,query_5,query_6,query_7,query_8,query_9,query_10,query_11,query_12,query_13,query_14,query_15,query_16,query_17,query_18,query_19,query_20,query_21,query_22,query_23,query_24,query_25,query_26,query_27,query_28,query_29,query_30,query_31,query_32,query_33,query_34,query_35,query_36]

    result_dict = {}
    k=[]
    for i, support_i in enumerate(support_set):
        support_i=support_i / torch.norm(support_i, p='fro')   #벡터를 크기로 나누어주기(normalize) [512]
        support_i=torch.unsqueeze(support_i, 1)                #[512,1]
        trans_support_i=support_i.t()                          #[1,512]
        for j, query_j in enumerate(query_set):

            query_j=query_j / torch.norm(query_j, p='fro')         #[512]
            query_j=torch.unsqueeze(query_j, 1)                    #[512,1]

            r_i_j=torch.matmul(trans_support_i,query_j)                #[1]              총 36x36 개 r 나오게 해야함
        
            # 원하는것 : r_32 하면 스칼라값 하나가 나오는거
        
            name = f"r_{i+1}_{j+1}"
            result_dict[name] = r_i_j
            r_i_j=r_i_j.tolist() 
            k.append(r_i_j)

    R=np.array(k)

    R_q=R.reshape(36,6,6)                 #R_q.shape => (36,6,6)

    pre_p=R.reshape(6,36,6)
    R_p=np.transpose(pre_p,(1,2,0))       #R_p.shape => (36,6,6)

    R_p=torch.tensor(R_p)                 # numpy array 를 pytorch tensor 로 바꿔줌
    R_q=torch.tensor(R_q)           

    gap=nn.AvgPool2d(6)
    m_p=gap(R_p)
    m_q=gap(R_q)

    m_p=m_p.squeeze()
    m_p=m_p.type('torch.FloatTensor')

    m_q=m_q.squeeze()
    m_q=m_q.type('torch.FloatTensor')

    model = torch.nn.Sequential(
        nn.Linear(36, 6),
        nn.ReLU(),
        nn.Linear(6, 36)
    )

    w_p=model(m_p)
    w_q=model(m_q)
    #################################################################6_make_w

    w_p = w_p.type('torch.DoubleTensor')
    w_q = w_q.type('torch.DoubleTensor')

    bunmo_p=[0]
    A_p=[]

    for i in range (6):
        for j in range (6):
            p=torch.matmul(R_p[:,i,j],w_p)
            p=p.detach().numpy()
            temp=np.exp(p/0.025)
            bunmo_p=bunmo_p+temp                         # 하나씩 더해진 분모

    for i in range (6):
        for j in range (6):
            p=torch.matmul(R_p[:,i,j],w_p)
            p=p.detach().numpy()
            bunja=np.exp(p/0.025)
            A_p.append(bunja/bunmo_p)

    #print(A_p)
    #print(bunmo_p)

    ###################################################################################
    bunmo_q=[0]
    A_q=[]

    for i in range (6):
        for j in range (6):
            q=torch.matmul(R_q[:,i,j],w_q)
            q=q.detach().numpy()
            temp=np.exp(q/0.025)
            bunmo_q=bunmo_q+temp                         # 하나씩 더해진 분모

    for i in range (6):
        for j in range (6):
            q=torch.matmul(R_q[:,i,j],w_q)
            q=q.detach().numpy()
            bunja=np.exp(q/0.025)
            A_q.append(bunja/bunmo_q)

    #print(A_q)
    #print(bunmo_q)

    ###############################################################7 Ap Aq 구하기

    for x in range(36):
        A_p[x]=A_p[x]+1
        A_q[x]=A_q[x]+1
    
    # support_feature , A_p Elementwisely weight
    support_feature=support_feature.detach().numpy()
    query_feature=query_feature.detach().numpy()

    for i in range(6):
        for j in range(6):
            support_feature[:,i,j]=np.multiply(support_feature[:,i,j],A_p[j+(6*i)])
        
    for i in range(6):
        for j in range(6):
            query_feature[:,i,j]=np.multiply(query_feature[:,i,j],A_q[j+(6*i)])

    #print(support_feature.shape)
    #print(query_feature.shape)

    pbar_two=support_feature
    qbar_two=query_feature
    ##############################################################8_make_Pbar_Qbar_two

    image=final_query[0][0]                                  # 0-5 총 6개 해주어야함

    image = image.transpose(2,0,1)[np.newaxis]               #[1,3,84,84] 로 만들어줌
    image = torch.tensor(image)                              # tensor로 만들어줌
    image = image.type('torch.FloatTensor')                  # float들로 숫자 저장

    model=resnet12()                                         #resnet불러오기
    query_feature=model(image)                               #resnet에 집어넣기

    # 서폿5개 레스넷 더한거 나누기5    (첫번째클래스 서폿셋)                                              
    temp=np.zeros((1,512,6,6))                                   # 5개 다 더할 temp 만들기
    temp=torch.tensor(temp)
    temp=temp.type('torch.FloatTensor')


    for p in range(10,15):                                         # 5-10 , 10-15 , 15-20 , 20-25 해주어야함 
        one_support = all_support[p][0]
        one_support = one_support.transpose(2,0,1)[np.newaxis]   #[1,3,84,84] 로 만들어줌
        one_support = torch.tensor(one_support)                  # tensor로 만들어줌
        one_support = one_support.type('torch.FloatTensor') 
    
        model=resnet12()
        added_feature=model(one_support)
        temp=temp+added_feature                                  # 0-4 5개 support data feature 합
    


    support_feature=temp/5                                       # 5개 feature의 평균

    support_feature=support_feature.view([-1, 6, 6])             # torch.Size([512, 6, 6])
    query_feature=query_feature.view([-1, 6, 6])                 # torch.Size([512, 6, 6])

    support_1=support_feature[:,0,0]
    support_2=support_feature[:,1,0]
    support_3=support_feature[:,2,0]
    support_4=support_feature[:,3,0]
    support_5=support_feature[:,4,0]
    support_6=support_feature[:,5,0]

    support_7=support_feature[:,0,1]
    support_8=support_feature[:,1,1]
    support_9=support_feature[:,2,1]
    support_10=support_feature[:,3,1]
    support_11=support_feature[:,4,1]
    support_12=support_feature[:,5,1]

    support_13=support_feature[:,0,2]
    support_14=support_feature[:,1,2]
    support_15=support_feature[:,2,2]
    support_16=support_feature[:,3,2]
    support_17=support_feature[:,4,2]
    support_18=support_feature[:,5,2]

    support_19=support_feature[:,0,3]
    support_20=support_feature[:,1,3]
    support_21=support_feature[:,2,3]
    support_22=support_feature[:,3,3]
    support_23=support_feature[:,4,3]
    support_24=support_feature[:,5,3]

    support_25=support_feature[:,0,4]
    support_26=support_feature[:,1,4]
    support_27=support_feature[:,2,4]
    support_28=support_feature[:,3,4]
    support_29=support_feature[:,4,4]
    support_30=support_feature[:,5,4]

    support_31=support_feature[:,0,5]
    support_32=support_feature[:,1,5]
    support_33=support_feature[:,2,5]
    support_34=support_feature[:,3,5]
    support_35=support_feature[:,4,5]
    support_36=support_feature[:,5,5]

    #########################################################################
    query_1=query_feature[:,0,0]
    query_2=query_feature[:,1,0]
    query_3=query_feature[:,2,0]
    query_4=query_feature[:,3,0]
    query_5=query_feature[:,4,0]
    query_6=query_feature[:,5,0]

    query_7=query_feature[:,0,1]
    query_8=query_feature[:,1,1]
    query_9=query_feature[:,2,1]
    query_10=query_feature[:,3,1]
    query_11=query_feature[:,4,1]
    query_12=query_feature[:,5,1]

    query_13=query_feature[:,0,2]
    query_14=query_feature[:,1,2]
    query_15=query_feature[:,2,2]
    query_16=query_feature[:,3,2]
    query_17=query_feature[:,4,2]
    query_18=query_feature[:,5,2]

    query_19=query_feature[:,0,3]
    query_20=query_feature[:,1,3]
    query_21=query_feature[:,2,3]
    query_22=query_feature[:,3,3]
    query_23=query_feature[:,4,3]
    query_24=query_feature[:,5,3]

    query_25=query_feature[:,0,4]
    query_26=query_feature[:,1,4]
    query_27=query_feature[:,2,4]
    query_28=query_feature[:,3,4]
    query_29=query_feature[:,4,4]
    query_30=query_feature[:,5,4]

    query_31=query_feature[:,0,5]
    query_32=query_feature[:,1,5]
    query_33=query_feature[:,2,5]
    query_34=query_feature[:,3,5]
    query_35=query_feature[:,4,5]
    query_36=query_feature[:,5,5]

    ############################################################################
    support_set = [support_1,support_2,support_3,support_4,support_5,support_6,support_7,support_8, support_9,support_10,support_11,support_12,support_13,support_14,  support_15,support_16, support_17,support_18,support_19,support_20, support_21,support_22, support_23,support_24,support_25,support_26,support_27,support_28, support_29,support_30,support_31,support_32,support_33,support_34,support_35,support_36]
    ############################################################################
    query_set = [query_1,query_2,query_3,query_4,query_5,query_6,query_7,query_8,query_9,query_10,query_11,query_12,query_13,query_14,query_15,query_16,query_17,query_18,query_19,query_20,query_21,query_22,query_23,query_24,query_25,query_26,query_27,query_28,query_29,query_30,query_31,query_32,query_33,query_34,query_35,query_36]

    result_dict = {}
    k=[]
    for i, support_i in enumerate(support_set):
        support_i=support_i / torch.norm(support_i, p='fro')   #벡터를 크기로 나누어주기(normalize) [512]
        support_i=torch.unsqueeze(support_i, 1)                #[512,1]
        trans_support_i=support_i.t()                          #[1,512]
        for j, query_j in enumerate(query_set):

            query_j=query_j / torch.norm(query_j, p='fro')         #[512]
            query_j=torch.unsqueeze(query_j, 1)                    #[512,1]

            r_i_j=torch.matmul(trans_support_i,query_j)                #[1]              총 36x36 개 r 나오게 해야함
        
            # 원하는것 : r_32 하면 스칼라값 하나가 나오는거
        
            name = f"r_{i+1}_{j+1}"
            result_dict[name] = r_i_j
            r_i_j=r_i_j.tolist() 
            k.append(r_i_j)

    R=np.array(k)

    R_q=R.reshape(36,6,6)                 #R_q.shape => (36,6,6)

    pre_p=R.reshape(6,36,6)
    R_p=np.transpose(pre_p,(1,2,0))       #R_p.shape => (36,6,6)

    R_p=torch.tensor(R_p)                 # numpy array 를 pytorch tensor 로 바꿔줌
    R_q=torch.tensor(R_q)           

    gap=nn.AvgPool2d(6)
    m_p=gap(R_p)
    m_q=gap(R_q)

    m_p=m_p.squeeze()
    m_p=m_p.type('torch.FloatTensor')

    m_q=m_q.squeeze()
    m_q=m_q.type('torch.FloatTensor')

    model = torch.nn.Sequential(
        nn.Linear(36, 6),
        nn.ReLU(),
        nn.Linear(6, 36)
    )

    w_p=model(m_p)
    w_q=model(m_q)
    #################################################################6_make_w

    w_p = w_p.type('torch.DoubleTensor')
    w_q = w_q.type('torch.DoubleTensor')

    bunmo_p=[0]
    A_p=[]

    for i in range (6):
        for j in range (6):
            p=torch.matmul(R_p[:,i,j],w_p)
            p=p.detach().numpy()
            temp=np.exp(p/0.025)
            bunmo_p=bunmo_p+temp                         # 하나씩 더해진 분모

    for i in range (6):
        for j in range (6):
            p=torch.matmul(R_p[:,i,j],w_p)
            p=p.detach().numpy()
            bunja=np.exp(p/0.025)
            A_p.append(bunja/bunmo_p)

    #print(A_p)
    #print(bunmo_p)

    ###################################################################################
    bunmo_q=[0]
    A_q=[]

    for i in range (6):
        for j in range (6):
            q=torch.matmul(R_q[:,i,j],w_q)
            q=q.detach().numpy()
            temp=np.exp(q/0.025)
            bunmo_q=bunmo_q+temp                         # 하나씩 더해진 분모

    for i in range (6):
        for j in range (6):
            q=torch.matmul(R_q[:,i,j],w_q)
            q=q.detach().numpy()
            bunja=np.exp(q/0.025)
            A_q.append(bunja/bunmo_q)

    #print(A_q)
    #print(bunmo_q)

    ###############################################################7 Ap Aq 구하기

    for x in range(36):
        A_p[x]=A_p[x]+1
        A_q[x]=A_q[x]+1
    
    # support_feature , A_p Elementwisely weight
    support_feature=support_feature.detach().numpy()
    query_feature=query_feature.detach().numpy()

    for i in range(6):
        for j in range(6):
            support_feature[:,i,j]=np.multiply(support_feature[:,i,j],A_p[j+(6*i)])
        
    for i in range(6):
        for j in range(6):
            query_feature[:,i,j]=np.multiply(query_feature[:,i,j],A_q[j+(6*i)])

    #print(support_feature.shape)
    #print(query_feature.shape)

    pbar_three=support_feature
    qbar_three=query_feature
    ##############################################################8_make_Pbar_Qbar_three

    image=final_query[0][0]                                  # 0-5 총 6개 해주어야함

    image = image.transpose(2,0,1)[np.newaxis]               #[1,3,84,84] 로 만들어줌
    image = torch.tensor(image)                              # tensor로 만들어줌
    image = image.type('torch.FloatTensor')                  # float들로 숫자 저장

    model=resnet12()                                         #resnet불러오기
    query_feature=model(image)                               #resnet에 집어넣기

    # 서폿5개 레스넷 더한거 나누기5    (첫번째클래스 서폿셋)                                              
    temp=np.zeros((1,512,6,6))                                   # 5개 다 더할 temp 만들기
    temp=torch.tensor(temp)
    temp=temp.type('torch.FloatTensor')


    for p in range(15,20):                                         # 5-10 , 10-15 , 15-20 , 20-25 해주어야함 
        one_support = all_support[p][0]
        one_support = one_support.transpose(2,0,1)[np.newaxis]   #[1,3,84,84] 로 만들어줌
        one_support = torch.tensor(one_support)                  # tensor로 만들어줌
        one_support = one_support.type('torch.FloatTensor') 
    
        model=resnet12()
        added_feature=model(one_support)
        temp=temp+added_feature                                  # 0-4 5개 support data feature 합
    


    support_feature=temp/5                                       # 5개 feature의 평균

    support_feature=support_feature.view([-1, 6, 6])             # torch.Size([512, 6, 6])
    query_feature=query_feature.view([-1, 6, 6])                 # torch.Size([512, 6, 6])

    support_1=support_feature[:,0,0]
    support_2=support_feature[:,1,0]
    support_3=support_feature[:,2,0]
    support_4=support_feature[:,3,0]
    support_5=support_feature[:,4,0]
    support_6=support_feature[:,5,0]

    support_7=support_feature[:,0,1]
    support_8=support_feature[:,1,1]
    support_9=support_feature[:,2,1]
    support_10=support_feature[:,3,1]
    support_11=support_feature[:,4,1]
    support_12=support_feature[:,5,1]

    support_13=support_feature[:,0,2]
    support_14=support_feature[:,1,2]
    support_15=support_feature[:,2,2]
    support_16=support_feature[:,3,2]
    support_17=support_feature[:,4,2]
    support_18=support_feature[:,5,2]

    support_19=support_feature[:,0,3]
    support_20=support_feature[:,1,3]
    support_21=support_feature[:,2,3]
    support_22=support_feature[:,3,3]
    support_23=support_feature[:,4,3]
    support_24=support_feature[:,5,3]

    support_25=support_feature[:,0,4]
    support_26=support_feature[:,1,4]
    support_27=support_feature[:,2,4]
    support_28=support_feature[:,3,4]
    support_29=support_feature[:,4,4]
    support_30=support_feature[:,5,4]

    support_31=support_feature[:,0,5]
    support_32=support_feature[:,1,5]
    support_33=support_feature[:,2,5]
    support_34=support_feature[:,3,5]
    support_35=support_feature[:,4,5]
    support_36=support_feature[:,5,5]

    #########################################################################
    query_1=query_feature[:,0,0]
    query_2=query_feature[:,1,0]
    query_3=query_feature[:,2,0]
    query_4=query_feature[:,3,0]
    query_5=query_feature[:,4,0]
    query_6=query_feature[:,5,0]

    query_7=query_feature[:,0,1]
    query_8=query_feature[:,1,1]
    query_9=query_feature[:,2,1]
    query_10=query_feature[:,3,1]
    query_11=query_feature[:,4,1]
    query_12=query_feature[:,5,1]

    query_13=query_feature[:,0,2]
    query_14=query_feature[:,1,2]
    query_15=query_feature[:,2,2]
    query_16=query_feature[:,3,2]
    query_17=query_feature[:,4,2]
    query_18=query_feature[:,5,2]

    query_19=query_feature[:,0,3]
    query_20=query_feature[:,1,3]
    query_21=query_feature[:,2,3]
    query_22=query_feature[:,3,3]
    query_23=query_feature[:,4,3]
    query_24=query_feature[:,5,3]

    query_25=query_feature[:,0,4]
    query_26=query_feature[:,1,4]
    query_27=query_feature[:,2,4]
    query_28=query_feature[:,3,4]
    query_29=query_feature[:,4,4]
    query_30=query_feature[:,5,4]

    query_31=query_feature[:,0,5]
    query_32=query_feature[:,1,5]
    query_33=query_feature[:,2,5]
    query_34=query_feature[:,3,5]
    query_35=query_feature[:,4,5]
    query_36=query_feature[:,5,5]

    ############################################################################
    support_set = [support_1,support_2,support_3,support_4,support_5,support_6,support_7,support_8, support_9,support_10,support_11,support_12,support_13,support_14,  support_15,support_16, support_17,support_18,support_19,support_20, support_21,support_22, support_23,support_24,support_25,support_26,support_27,support_28, support_29,support_30,support_31,support_32,support_33,support_34,support_35,support_36]
    ############################################################################
    query_set = [query_1,query_2,query_3,query_4,query_5,query_6,query_7,query_8,query_9,query_10,query_11,query_12,query_13,query_14,query_15,query_16,query_17,query_18,query_19,query_20,query_21,query_22,query_23,query_24,query_25,query_26,query_27,query_28,query_29,query_30,query_31,query_32,query_33,query_34,query_35,query_36]

    result_dict = {}
    k=[]
    for i, support_i in enumerate(support_set):
        support_i=support_i / torch.norm(support_i, p='fro')   #벡터를 크기로 나누어주기(normalize) [512]
        support_i=torch.unsqueeze(support_i, 1)                #[512,1]
        trans_support_i=support_i.t()                          #[1,512]
        for j, query_j in enumerate(query_set):

            query_j=query_j / torch.norm(query_j, p='fro')         #[512]
            query_j=torch.unsqueeze(query_j, 1)                    #[512,1]

            r_i_j=torch.matmul(trans_support_i,query_j)                #[1]              총 36x36 개 r 나오게 해야함
        
            # 원하는것 : r_32 하면 스칼라값 하나가 나오는거
        
            name = f"r_{i+1}_{j+1}"
            result_dict[name] = r_i_j
            r_i_j=r_i_j.tolist() 
            k.append(r_i_j)

    R=np.array(k)

    R_q=R.reshape(36,6,6)                 #R_q.shape => (36,6,6)

    pre_p=R.reshape(6,36,6)
    R_p=np.transpose(pre_p,(1,2,0))       #R_p.shape => (36,6,6)

    R_p=torch.tensor(R_p)                 # numpy array 를 pytorch tensor 로 바꿔줌
    R_q=torch.tensor(R_q)           

    gap=nn.AvgPool2d(6)
    m_p=gap(R_p)
    m_q=gap(R_q)

    m_p=m_p.squeeze()
    m_p=m_p.type('torch.FloatTensor')

    m_q=m_q.squeeze()
    m_q=m_q.type('torch.FloatTensor')

    model = torch.nn.Sequential(
        nn.Linear(36, 6),
        nn.ReLU(),
        nn.Linear(6, 36)
    )

    w_p=model(m_p)
    w_q=model(m_q)
    #################################################################6_make_w

    w_p = w_p.type('torch.DoubleTensor')
    w_q = w_q.type('torch.DoubleTensor')

    bunmo_p=[0]
    A_p=[]

    for i in range (6):
        for j in range (6):
            p=torch.matmul(R_p[:,i,j],w_p)
            p=p.detach().numpy()
            temp=np.exp(p/0.025)
            bunmo_p=bunmo_p+temp                         # 하나씩 더해진 분모

    for i in range (6):
        for j in range (6):
            p=torch.matmul(R_p[:,i,j],w_p)
            p=p.detach().numpy()
            bunja=np.exp(p/0.025)
            A_p.append(bunja/bunmo_p)

    #print(A_p)
    #print(bunmo_p)

    ###################################################################################
    bunmo_q=[0]
    A_q=[]

    for i in range (6):
        for j in range (6):
            q=torch.matmul(R_q[:,i,j],w_q)
            q=q.detach().numpy()
            temp=np.exp(q/0.025)
            bunmo_q=bunmo_q+temp                         # 하나씩 더해진 분모

    for i in range (6):
        for j in range (6):
            q=torch.matmul(R_q[:,i,j],w_q)
            q=q.detach().numpy()
            bunja=np.exp(q/0.025)
            A_q.append(bunja/bunmo_q)

    #print(A_q)
    #print(bunmo_q)

    ###############################################################7 Ap Aq 구하기

    for x in range(36):
        A_p[x]=A_p[x]+1
        A_q[x]=A_q[x]+1
    
    # support_feature , A_p Elementwisely weight
    support_feature=support_feature.detach().numpy()
    query_feature=query_feature.detach().numpy()

    for i in range(6):
        for j in range(6):
            support_feature[:,i,j]=np.multiply(support_feature[:,i,j],A_p[j+(6*i)])
        
    for i in range(6):
        for j in range(6):
            query_feature[:,i,j]=np.multiply(query_feature[:,i,j],A_q[j+(6*i)])

    #print(support_feature.shape)
    #print(query_feature.shape)

    pbar_four=support_feature
    qbar_four=query_feature
    ##############################################################8_make_Pbar_Qbar_four

    image=final_query[0][0]                                  # 0-5 총 6개 해주어야함

    image = image.transpose(2,0,1)[np.newaxis]               #[1,3,84,84] 로 만들어줌
    image = torch.tensor(image)                              # tensor로 만들어줌
    image = image.type('torch.FloatTensor')                  # float들로 숫자 저장

    model=resnet12()                                         #resnet불러오기
    query_feature=model(image)                               #resnet에 집어넣기

    # 서폿5개 레스넷 더한거 나누기5    (첫번째클래스 서폿셋)                                              
    temp=np.zeros((1,512,6,6))                                   # 5개 다 더할 temp 만들기
    temp=torch.tensor(temp)
    temp=temp.type('torch.FloatTensor')


    for p in range(5,10):                                         # 5-10 , 10-15 , 15-20 , 20-25 해주어야함 
        one_support = all_support[p][0]
        one_support = one_support.transpose(2,0,1)[np.newaxis]   #[1,3,84,84] 로 만들어줌
        one_support = torch.tensor(one_support)                  # tensor로 만들어줌
        one_support = one_support.type('torch.FloatTensor') 
    
        model=resnet12()
        added_feature=model(one_support)
        temp=temp+added_feature                                  # 0-4 5개 support data feature 합
    


    support_feature=temp/5                                       # 5개 feature의 평균

    support_feature=support_feature.view([-1, 6, 6])             # torch.Size([512, 6, 6])
    query_feature=query_feature.view([-1, 6, 6])                 # torch.Size([512, 6, 6])

    support_1=support_feature[:,0,0]
    support_2=support_feature[:,1,0]
    support_3=support_feature[:,2,0]
    support_4=support_feature[:,3,0]
    support_5=support_feature[:,4,0]
    support_6=support_feature[:,5,0]

    support_7=support_feature[:,0,1]
    support_8=support_feature[:,1,1]
    support_9=support_feature[:,2,1]
    support_10=support_feature[:,3,1]
    support_11=support_feature[:,4,1]
    support_12=support_feature[:,5,1]

    support_13=support_feature[:,0,2]
    support_14=support_feature[:,1,2]
    support_15=support_feature[:,2,2]
    support_16=support_feature[:,3,2]
    support_17=support_feature[:,4,2]
    support_18=support_feature[:,5,2]

    support_19=support_feature[:,0,3]
    support_20=support_feature[:,1,3]
    support_21=support_feature[:,2,3]
    support_22=support_feature[:,3,3]
    support_23=support_feature[:,4,3]
    support_24=support_feature[:,5,3]

    support_25=support_feature[:,0,4]
    support_26=support_feature[:,1,4]
    support_27=support_feature[:,2,4]
    support_28=support_feature[:,3,4]
    support_29=support_feature[:,4,4]
    support_30=support_feature[:,5,4]

    support_31=support_feature[:,0,5]
    support_32=support_feature[:,1,5]
    support_33=support_feature[:,2,5]
    support_34=support_feature[:,3,5]
    support_35=support_feature[:,4,5]
    support_36=support_feature[:,5,5]

    #########################################################################
    query_1=query_feature[:,0,0]
    query_2=query_feature[:,1,0]
    query_3=query_feature[:,2,0]
    query_4=query_feature[:,3,0]
    query_5=query_feature[:,4,0]
    query_6=query_feature[:,5,0]

    query_7=query_feature[:,0,1]
    query_8=query_feature[:,1,1]
    query_9=query_feature[:,2,1]
    query_10=query_feature[:,3,1]
    query_11=query_feature[:,4,1]
    query_12=query_feature[:,5,1]

    query_13=query_feature[:,0,2]
    query_14=query_feature[:,1,2]
    query_15=query_feature[:,2,2]
    query_16=query_feature[:,3,2]
    query_17=query_feature[:,4,2]
    query_18=query_feature[:,5,2]

    query_19=query_feature[:,0,3]
    query_20=query_feature[:,1,3]
    query_21=query_feature[:,2,3]
    query_22=query_feature[:,3,3]
    query_23=query_feature[:,4,3]
    query_24=query_feature[:,5,3]

    query_25=query_feature[:,0,4]
    query_26=query_feature[:,1,4]
    query_27=query_feature[:,2,4]
    query_28=query_feature[:,3,4]
    query_29=query_feature[:,4,4]
    query_30=query_feature[:,5,4]

    query_31=query_feature[:,0,5]
    query_32=query_feature[:,1,5]
    query_33=query_feature[:,2,5]
    query_34=query_feature[:,3,5]
    query_35=query_feature[:,4,5]
    query_36=query_feature[:,5,5]

    ############################################################################
    support_set = [support_1,support_2,support_3,support_4,support_5,support_6,support_7,support_8, support_9,support_10,support_11,support_12,support_13,support_14,  support_15,support_16, support_17,support_18,support_19,support_20, support_21,support_22, support_23,support_24,support_25,support_26,support_27,support_28, support_29,support_30,support_31,support_32,support_33,support_34,support_35,support_36]
    ############################################################################
    query_set = [query_1,query_2,query_3,query_4,query_5,query_6,query_7,query_8,query_9,query_10,query_11,query_12,query_13,query_14,query_15,query_16,query_17,query_18,query_19,query_20,query_21,query_22,query_23,query_24,query_25,query_26,query_27,query_28,query_29,query_30,query_31,query_32,query_33,query_34,query_35,query_36]

    result_dict = {}
    k=[]
    for i, support_i in enumerate(support_set):
        support_i=support_i / torch.norm(support_i, p='fro')   #벡터를 크기로 나누어주기(normalize) [512]
        support_i=torch.unsqueeze(support_i, 1)                #[512,1]
        trans_support_i=support_i.t()                          #[1,512]
        for j, query_j in enumerate(query_set):

            query_j=query_j / torch.norm(query_j, p='fro')         #[512]
            query_j=torch.unsqueeze(query_j, 1)                    #[512,1]

            r_i_j=torch.matmul(trans_support_i,query_j)                #[1]              총 36x36 개 r 나오게 해야함
        
            # 원하는것 : r_32 하면 스칼라값 하나가 나오는거
        
            name = f"r_{i+1}_{j+1}"
            result_dict[name] = r_i_j
            r_i_j=r_i_j.tolist() 
            k.append(r_i_j)

    R=np.array(k)

    R_q=R.reshape(36,6,6)                 #R_q.shape => (36,6,6)

    pre_p=R.reshape(6,36,6)
    R_p=np.transpose(pre_p,(1,2,0))       #R_p.shape => (36,6,6)

    R_p=torch.tensor(R_p)                 # numpy array 를 pytorch tensor 로 바꿔줌
    R_q=torch.tensor(R_q)           

    gap=nn.AvgPool2d(6)
    m_p=gap(R_p)
    m_q=gap(R_q)

    m_p=m_p.squeeze()
    m_p=m_p.type('torch.FloatTensor')

    m_q=m_q.squeeze()
    m_q=m_q.type('torch.FloatTensor')

    model = torch.nn.Sequential(
        nn.Linear(36, 6),
        nn.ReLU(),
        nn.Linear(6, 36)
    )

    w_p=model(m_p)
    w_q=model(m_q)
    #################################################################6_make_w

    w_p = w_p.type('torch.DoubleTensor')
    w_q = w_q.type('torch.DoubleTensor')

    bunmo_p=[0]
    A_p=[]

    for i in range (6):
        for j in range (6):
            p=torch.matmul(R_p[:,i,j],w_p)
            p=p.detach().numpy()
            temp=np.exp(p/0.025)
            bunmo_p=bunmo_p+temp                         # 하나씩 더해진 분모

    for i in range (6):
        for j in range (6):
            p=torch.matmul(R_p[:,i,j],w_p)
            p=p.detach().numpy()
            bunja=np.exp(p/0.025)
            A_p.append(bunja/bunmo_p)

    #print(A_p)
    #print(bunmo_p)

    ###################################################################################
    bunmo_q=[0]
    A_q=[]

    for i in range (6):
        for j in range (6):
            q=torch.matmul(R_q[:,i,j],w_q)
            q=q.detach().numpy()
            temp=np.exp(q/0.025)
            bunmo_q=bunmo_q+temp                         # 하나씩 더해진 분모

    for i in range (6):
        for j in range (6):
            q=torch.matmul(R_q[:,i,j],w_q)
            q=q.detach().numpy()
            bunja=np.exp(q/0.025)
            A_q.append(bunja/bunmo_q)

    #print(A_q)
    #print(bunmo_q)

    ###############################################################7 Ap Aq 구하기

    for x in range(36):
        A_p[x]=A_p[x]+1
        A_q[x]=A_q[x]+1
    
    # support_feature , A_p Elementwisely weight
    support_feature=support_feature.detach().numpy()
    query_feature=query_feature.detach().numpy()

    for i in range(6):
        for j in range(6):
            support_feature[:,i,j]=np.multiply(support_feature[:,i,j],A_p[j+(6*i)])
        
    for i in range(6):
        for j in range(6):
            query_feature[:,i,j]=np.multiply(query_feature[:,i,j],A_q[j+(6*i)])

    #print(support_feature.shape)
    #print(query_feature.shape)

    pbar_five=support_feature
    qbar_five=query_feature
    ##############################################################8_make_Pbar_Qbar_five
    return pbar_one, pbar_two, pbar_three, pbar_four, pbar_five, qbar_one, qbar_two, qbar_three, qbar_four, qbar_five, real_query_class, class_five