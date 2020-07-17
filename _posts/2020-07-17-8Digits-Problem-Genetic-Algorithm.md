---
key: 2020-07-17-8Digits-Problem-Genetic-Algorithm.md
show_edit_on_github: false
title: 遗传算法解决8数码问题
tags: 算法 编程 Python 
---

# 8数码问题描述  
3×3九宫格，将数字1-8放入宫格内，剩余一个空位，可通过将数字滑动至空格改变布局，给定一个初始状态，在有限步内将其转换为目标状态。
|   |   |   |
|:-:|:-:|:-:|
| 7 | 3 | X |
| 8 | 1 | 4 |
| 6 | 5 | 2 |  
<!--more-->
上图就是一个8数码问题，X表示空位，经过「右 上 上 右 下 下 左 上 左 上 右 右 下 左 左 上」一系列滑动操作即可得到下图状态。
|   |   |   |
|:-:|:-:|:-:|
| 1 | 2 | 3 |
| 4 | 5 | 6 |
| 7 | 8 | X |  
遗传算法的求解目标即是这一系列滑动操作。

# 遗传算法简介  
遗传算法是一种通过模拟自然进化过程搜索最优解的方法。该算法通过数学的方式,利用计算机仿真运算,将问题的求解过程转换成类似生物进化中的染色体基因的交叉、变异等过程。在求解较为复杂的组合优化问题时,相对一些常规的优化算法,通常能够较快地获得较好的优化结果[^1]  
![遗传算法流程图](https://bkimg.cdn.bcebos.com/pic/8ad4b31c8701a18b87d61b662167100828381f3050cb "遗传算法流程图")  
算法基本思路如上图所示  

# 编程环境——Jupyter Notebook  
![Jupyter Notebook](https://jupyter.org/assets/main-logo.svg "Jupyter Notebook")  
Jupyter Notebook（此前被称为 IPython notebook）是一个交互式笔记本，支持运行 40 多种编程语言。  
Jupyter Notebook 的本质是一个 Web 应用程序，便于创建和共享文学化程序文档，支持实时代码，数学方程，可视化和 markdown。 用途包括：数据清理和转换，数值模拟，统计建模，机器学习等等。  
[Jupyter Notebook 安装&配置](https://jupyter.org/install)  

# 实现思路  
首先，将移动策略抽象为「基因」，通过按照策略移动后的状态与目标状态的差距作为评价策略好坏的权值（权值具体算法见下部分），以此作为杂交、变异以及淘汰的依据。  
实际运行时，随机生成含有若干策略的「策略组」，具体策略的长度可通过超参数自定义，然后经过若干次迭代，每次迭代都包括杂交、变异、淘汰这三部分，如果迭代过程中产生了权值为0的策略则输出并结束程序，否则输出当前最优策略，并将这个策略的优化版本作为第二轮迭代策略组的起始操作步骤，后续继续随机生成，进入第二次大迭代。  


# 具体代码  
## 1. 单次移动  
```python
def move_once(inital_state, operation):
    col_size = board_shape[1]
    state = inital_state.flatten()
    zero_pos = np.argmin(state)
    
    #up
    if operation==0 and zero_pos<(state.size - col_size):
        state[zero_pos], state[zero_pos+col_size] = state[zero_pos+col_size], state[zero_pos]
    #right
    elif operation==1 and zero_pos%col_size!=0:
        state[zero_pos], state[zero_pos-1] = state[zero_pos-1], state[zero_pos]
    #down    
    elif operation==2 and zero_pos>=col_size:
        state[zero_pos], state[zero_pos-col_size] = state[zero_pos-col_size], state[zero_pos]
    #left
    elif operation==3 and zero_pos%col_size!=col_size-1:
        state[zero_pos], state[zero_pos+1] = state[zero_pos+1], state[zero_pos]

    return state.reshape(inital_state.shape)
```
实现移动是改变初始状态的基本条件，这里将问题中的九宫格定义为一个3x3的矩阵数组（函数过程中为了方便操作会转换为1x9的数组），空位定义为0存储在数组中。  
移动操作分为「上、下、左、右」四个方向，从数据结构上理解，操作本质就是0和其他数字交换位置。  
特别要考虑的是无效操作，以下图为例：  
|   |   |   |
|:-:|:-:|:-:|
| 7 | 3 | 0 |
| 8 | 1 | 4 |
| 6 | 5 | 2 |  
此时「下，左」两种操作是无法改变棋盘状态的，需要忽略不计。  
函数参数有两个，第一个是矩阵数组，即某个期盼状态，第二个参数是某个操作，这里用整数表示，「0，1，2，3」对应「上 右 下 左」，返回值是经过这个操作移动后的棋盘状态对应的矩阵数组。  

## 2. 按策略多次移动  
```python
def move_with_policy(state, policy_moves):
    for i in range(len(policy_moves)):
        state = move_once(state, policy_moves[i])
    
    return state
```
通过对单次移动的多次调用，实现了对一系列操作后棋盘状态的获取。  
参数和上面move_once函数类似，只是第二个参数为一个包含了若干个「0，1，2，3」的数组。  

## 3. 状态权值计算  
```python
def valueOf(state):
    value = 0
    #通过数字坐标差的模长来衡量和目标状态的差距
    for i in range(board_shape[0]*board_shape[1]):
        value += np.linalg.norm(np.argwhere(state==i)-np.argwhere(target==i))
    
    return value
```
通过计算中间状态每个数字坐标和其对应的目标状态坐标差向量的模长，来衡量该数字距离正确位置的差距，一个中间状态的权值即为0-8这9个数字的权值之和，当中间状态达到目标状态时，该状态权值为0。  

## 4. 杂交  
```python
#针对一个策略组杂交，返回杂交后的新策略组
def hybrid(policies):
    #按value从小到大排列
    policies = np.sort(policies, order = 'value')
    #计算杂交数量
    hybrid_count = np.around(len(policies)*hybrid_rate/100).astype(int)
    
    #value排名在前 hybrid_rate 的策略随机和策略组中任意一个杂交
    hybrid_index = np.random.choice(np.arange(len(policies)), hybrid_count)
    policies = np.append(policies, hybrid_2(policies[:hybrid_count], policies[hybrid_index]))
    policies = np.sort(policies, order = 'value')
    
    return policies if len(policies)<policy_limit else policies[:policy_limit]

#针对两组策略一一杂交
def hybrid_2(policies1, policies2):
    for i in range(len(policies1)):
        cut_pos = np.random.choice(np.arange(1,move_count-1))
        policies1[i]['moves'] = np.hstack((policies1[i]['moves'][:cut_pos], policies2[i]['moves'][cut_pos:]))
        policies1[i]['value'] = valueOf(move_with_policy(inital, policies[i]['moves']))
    
    return policies1
```
这里将「杂交」定义为将两个等长的策略移动步骤从相同位置断开，并将前者的前半部分和后者的后半部分拼接，形成一个新的移动步骤，并计算新步骤的状态权值，从而产生一个新策略。  
要注意的是，对于8数码问题来说，两个优秀的策略杂交，由于断开位置不同，可能并不能得到一个更加优秀的策略。  
|   |   |   |
|:-:|:-:|:-:|
| 7 | 3 | 0 |
| 8 | 1 | 4 |
| 6 | 5 | 2 |

|   |   |   |
|:-:|:-:|:-:|
| 7 | 0 | 3 |
| 8 | 1 | 4 |
| 6 | 5 | 2 |  
以上面这样的状态变换为例，可通过策略「下左下右」和「下右下下」得到，移动后新状态权值约为14.7。但如果这两个策略杂交生成「下左下下」这个新策略，对于初始状态其实并没有产生实际操作，状态权值约为15.5。  

## 5. 变异  
```python
def mutate(policies):
    for i in range(len(policies)):
        policy_original = policies[i]
        if(np.random.random() * 100 < mutate_rate):
            policies[i]['moves'][np.random.choice(range(move_count))] = np.random.choice(range(direct_count))
            policies[i]['value'] = valueOf(move_with_policy(inital, policies[i]['moves']))
            #淘汰劣势变异
            if better_mutate and policies[i]['value'] > policy_original['value']:
                policies[i] = policy_original
    
    return np.sort(policies, order = 'value')
```
变异实现较为简单，通过超参数设置变异概率，通过随机生成0-1之间的浮点数判断是否小于变异概率，来代表变异是否发生。  
变异发生即将策略随机某个位置的移动方向改变，即修改策略整型数组中随机下标位置的一个值。可以先保留变异前的策略，如果变异后策略权值变大，即远离目标状态，则可以保留变异前的策略，可理解为一次失败变异。  

## 6. 策略优化  
```python
def shorten(policy_editable):
    #防止修改policy原来的值
    policy = copy.deepcopy(policy_editable)
    policy_moves = shorten1(policy['moves'])
    short_policy_moves = shorten2(policy_moves)
    while len(policy_moves)!=len(short_policy_moves):
        policy_moves = short_policy_moves
        short_policy_moves = shorten2(short_policy_moves)
        
    return short_policy_moves
        
#去除无效操作
def shorten1(policy_moves_editable):
    #防止修改policy原来的值
    policy_moves = copy.deepcopy(policy_moves_editable)
    state = inital
    for i in range(len(policy_moves)):
        state_new = move_once(state, policy_moves[i])
        if np.array_equal(state_new, state):
            policy_moves[i] = -1
        else:
            state = state_new
    
    return np.delete(policy_moves, np.argwhere(policy_moves == -1))

#去除往复操作
def shorten2(policy_moves_editable):
    #防止修改policy原来的值
    policy_moves = copy.deepcopy(policy_moves_editable)
    for i in range(len(policy_moves)-1):
        if abs(policy_moves[i]-policy_moves[i+1])==(int)(direct_count/2) and policy_moves[i]!=-1:
            policy_moves[i] = policy_moves[i+1] = -1
            ++i
    
    return np.delete(policy_moves, np.argwhere(policy_moves == -1))
```
随机生成的移动策略可能包含「不能改变宫格状态」的操作，所以我们可以通过逐步执行移动策略同时比较前后宫格变化来过滤掉无效操作。  
在去除掉无效操作后，移动策略中还可能包含往复操作，例如：「上下」、「左左右右」，也可忽略不计，从而缩短移动策略步数。  

[^1]:[遗传算法——百度百科](https://baike.baidu.com/item/%E9%81%97%E4%BC%A0%E7%AE%97%E6%B3%95)  
