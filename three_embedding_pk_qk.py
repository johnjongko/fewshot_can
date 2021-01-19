def embedding_pk_qk():

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