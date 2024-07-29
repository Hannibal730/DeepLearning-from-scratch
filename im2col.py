# p.245 생략된 im2col 주석달며 공부하기 


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    
    N, C, H, W = input_data.shape # 각 변수에 할당
    
    out_h = (H + 2*pad - filter_h)//stride + 1 # 패딩마친 이미지가 필터를 거쳤을 때 출력 이미지의 높이 p.234
    out_w = (W + 2*pad - filter_w)//stride + 1 # 출력 이미지의 너비
    # 참고로 수직방향으로는 out_h만큼 필터가 작동한 것이다.
    # 작동 한 번 할 때에 결과가 하나 나오니까, out_h만큼 결과가 나온 것은 out_h만큼 작동했다는 것.
    # 그리고 맨 처음에는 이동 안 해도 작동하니까, out_h만큼 결과가 나온 것은 out_h-1번 이동한 것.
    

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    # (인풋배치 개수, 채널 수, 하이트, 윗스)형태였던 인풋데이터에 패딩할 거임.
    # 인풋배치 개수, 채널 수는 각각 (0,0) 즉 0칸씩 패딩하고 (패딩 안 한다는 의미)
    # 하이트는 (pad,pad) 즉 첫 행 위에 pad칸씩 패딩, 마지막행 위에 pad칸씩 패딩
    # 윗스는 (pad,pad) 즉 첫 열 왼쪽에 pad칸씩 패딩, 마지막 열 왼쪽에 pad칸씩 패딩
    # 컨스턴트 형식으로 세팅만 해둔 상황인데, 컨스턴트 형식일 때 패딩되는 기본값은 0이다


    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
    # im2col로 2차원 배열로 변환   
    # 일단 차원의 사이즈만 맞춘 채 값들은 0으로
    
    for y in range(filter_h):
        
        y_max = y + stride*out_h
        # 위에서 필터 사이즈를 입력했고, 그로부터 out 사이즈를 계산해서 알게 됐다.
        # 또한 입력데이터의 사이즈도 입력해서 알고 있다.
        # y는 range(필터사이즈)이다.
        # y_max는 입력데이터에서 필터가 커버가능한 끝부분-1이다. **슬라이싱할 때 쓰는 종료지점
        # 예를 들어서 입력사이즈가 5, 필터 사이즈가3, 스트라이드가1, 패딩0이면 계산했을 때 출력이 3이다.
        # 이때 y는 range(필터사이즈인3)이라서 [0,1,2]이다.
        # 그리고 y=0일 때 y_max를 계산해보면 3이 나오는데, 이는 입력데이터[0:3]을 의미하기 때문에 입력데이터의 0번째 행부터 2번째 행까지 필터가 커버하는 상황이다.
        # y=2일 때는 y_max는 5이고, 마찬가지로 이는 입력데이터[2:5]이고, 입력데이터의 2번째 행부터 4번째 행까지 필터가 커버한다.
        
        for x in range(filter_w):
            # 행이 선택됐다면, 그 행 안에서 첫 열부터 반복문 시작 (중첩 반복문)
            x_max = x + stride*out_w
            col [:,  :,  y,  x,  :,  :] = img [:,  :,  y:y_max: stride,  x:x_max : stride]
            # img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
            # 패딩이 완료된 이미지 데이터의 특징을 위에서 만든 영행렬에 인덱싱.
            # 패딩이 완료된 이미지 데이터에서 슬라이싱할 거야.
            # 어떻게? 하이트에  [y:y_max: stride]를 하면 시작점 y, 종료지점 y_max (아하! 이때 쓰려고 y_max가 슬라이싱 때 쓰는 종료지점처럼 계산된 거구나.) stride 간격으로 슬라이싱.
            
            # 중첩 반복문으로 형성되는 x와 y의 조합에 따라서, 매 조합 때마다 col에 저장됨.
            # 예를 들면
            # y = 0
            # y_max = y + stride * out_h = 0 + 1 * 3 = 3

            # x = 0
            # x_max = x + stride * out_w = 0 + 1 * 3 = 3

            # 이때의 col[:, :, 0, 0, :, :] = img[:, :, 0:3:1, 0:3:1]
            # 참고로 img[:, :, 0:3:1, 0:3:1]는 예를들어 
            # [[1, 2, 3],
            # [5, 6, 7],
            # [9, 10, 11]] 이런 형태인 느낌.
            
            # 결국 매 x,y조합마다 '해당 x,y를 좌상단 인덱싱값으로 갖는 필터'가 입력데이터를 cnn완료하고 출력한 배열이 저장됨.

            
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    # 6차원 배열을 2차원 배열하는 마법
    # transpose(0, 4, 5, 1, 2, 3)는 col 배열의 차원을 다음 순서로 재배열함
    # 기존 차원 순서: (N, C, filter_h, filter_w, out_h, out_w)
    #                0   1     2         3        4      5
    
    # 새로운 차원 순서: (N, out_h, out_w, C, filter_h, filter_w)
    
    # .reshape(N * out_h * out_w, -1)은
    #  잎의 배열을 (N * out_h * out_w, -1) 형태로 바꿈.
    #  -1은 자동으로 남은 차원 크기를 계산하여 맞춰주는 것을 의미함.
    
    # 예를 들어 col = (1, 1, 2, 2, 3, 3)
    # col.transposecol(0, 4, 5, 1, 2, 3) = (1, 3, 3, 1, 2, 2)
    # col.reshape = (9, 4)

    
    return col