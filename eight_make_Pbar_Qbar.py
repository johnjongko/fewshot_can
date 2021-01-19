def make_Pbar_Qbar():

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