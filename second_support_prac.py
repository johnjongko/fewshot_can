
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
support_set = [support_1,
            support_2,
            support_3,
            support_4,
            support_5,
            support_6,
            support_7,
            support_8,
            support_9,
            support_10,
            support_11,
            support_12,
            support_13,
            support_14,
            support_15,
            support_16,
            support_17,
            support_18,
            support_19,
            support_20,
            support_21,
            support_22,
            support_23,
            support_24,
            support_25,
            support_26,
            support_27,
            support_28,
            support_29,
            support_30,
            support_31,
            support_32,
            support_33,
            support_34,
            support_35,
            support_36]

############################################################################
query_set = [query_1,
            query_2,
            query_3,
            query_4,
            query_5,
            query_6,
            query_7,
            query_8,
            query_9,
            query_10,
            query_11,
            query_12,
            query_13,
            query_14,
            query_15,
            query_16,
            query_17,
            query_18,
            query_19,
            query_20,
            query_21,
            query_22,
            query_23,
            query_24,
            query_25,
            query_26,
            query_27,
            query_28,
            query_29,
            query_30,
            query_31,
            query_32,
            query_33,
            query_34,
            query_35,
            query_36]

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

print(A_p)
print(bunmo_p)

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

print(support_feature.shape)
print(query_feature.shape)
##############################################################8_make_Pbar_Qbar
