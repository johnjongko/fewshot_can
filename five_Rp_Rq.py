def Rp_Rq():

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


  