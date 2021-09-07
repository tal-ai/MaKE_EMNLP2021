import pandas as pd
import re
import sentencepiece as spm
import copy
import torch


def process_scene(scene):
    scene = scene.replace(' ','')
    map_scene = {}
    edge_0, edge_1 = re.split(r'\),|\)，',scene)
    ent0, ent1 = edge_0.split('(')
    map_scene[ent0] = ent1

    ent2, ent3 = edge_1[:-1].split('(')
    map_scene[ent2] = ent3
    return map_scene
def process_var(var):
    map_var = {}
    edge_0, edge_1 = re.split(r'\),|\)，',var)
    ent0, ent1 = edge_0.split('(')
    map_var[ent1] = ent0.replace(' ','')
    ent2, ent3 = edge_1[:-1].split('(')
    map_var[ent3] = ent2.replace(' ','')
    return map_var
def process_mr(mr, var):
    splited_mr = mr.split(',')
    mr_dict = {}
    for one in splited_mr:
        ent0, ent1 = one.split('[')
        mr_dict[ent0] = ent1[:-1]
    processed_var = process_var(var)
    mr_dict['x_ent'] = processed_var['x']
    mr_dict['y_ent'] = processed_var['y']
    return mr_dict
def fenge_B006(text):
    a = re.split(r'(\+|,|-|\*|/|=|!=|>=|<=|>|<|\(|\)|\[|\]|\\|\|\b\d+\w*\b|[\u4E00-\u9FA5]|\②|\①|\③|\④|\㉖|\:)', text)
    return [x.strip() for x in a if x!='']


train_set = pd.read_excel('./data/train_equ.xlsx', index_col=None)
dev_set = pd.read_excel('./data/dev_equ.xlsx')
test_set = pd.read_excel('./data/test_equ.xlsx')



def get_equation_mr(equation1, equation2):
    equ1_left = equation1[:equation1.find('=')]
    equ1_right = equation1[equation1.find('=')+1:]
    splited_left_1 = fenge_B006(equ1_left)
    splited_right_1 = fenge_B006(equ1_right)
    one_mr = []
    # 先处理第一个式子的左半部分
    if len(splited_left_1) == 3:
        # 处理x部分
        if splited_left_1[0] =='x':
            pass
        else:
            one_mr.append('eq1_x_index[{}]'.format(splited_left_1[0].replace('x','')))
        # 处理中间的那个运算符
        one_mr.append('eq1_left_sym2[{}]'.format(splited_left_1[1]))
        # 处理y部分
        if splited_left_1[2] =='y':
            pass
        else:
            one_mr.append('eq1_y_index[{}]'.format(splited_left_1[2].replace('y','')))
    elif len(splited_left_1) == 4:
        # 有四个部分，看起来都要做填充
        # 第一个部分的符号
        one_mr.append('eq1_left_sym1[{}]'.format(splited_left_1[0]))
        if splited_left_1[1] =='x':
            pass
        else:
            one_mr.append('eq1_x_index[{}]'.format(splited_left_1[1].replace('x','')))
        # 第二个部分的符号
        one_mr.append('eq1_left_sym2[{}]'.format(splited_left_1[2]))
        # 处理y部分
        if splited_left_1[3] =='y':
            pass
        else:
            one_mr.append('eq1_y_index[{}]'.format(splited_left_1[3].replace('y','')))
    else:
        raise 'wrong input equation1, check preprocessing'

    # 处理第一个式子的右半部分
    if len(splited_right_1) == 1:
        # 只有一个实数，应该大部分都是如此
        one_mr.append('eq1_right_num1[{}]'.format(splited_right_1[0]))
    elif len(splited_right_1) == 3:
        # 有实数的计算存在，好吧，那就算一算吧
        one_mr.append('eq1_right_num1[{}]'.format(splited_right_1[0]))
        one_mr.append('eq1_right_sym[{}]'.format(splited_right_1[1]))
        one_mr.append('eq1_right_num2[{}]'.format(splited_right_1[2]))


    equ2_left = equation2[:equation2.find('=')]
    equ2_right = equation2[equation2.find('=')+1:]
    splited_left_2 = fenge_B006(equ2_left)
    splited_right_2 = fenge_B006(equ2_right)

    # 先处理第二个式子的左半部分
    if len(splited_left_2) == 3:
        # 处理x部分
        if splited_left_2[0] =='x':
            pass
        else:
            one_mr.append('eq2_x_index[{}]'.format(splited_left_2[0].replace('x','')))
        # 处理中间的那个运算符
        one_mr.append('eq2_left_sym2[{}]'.format(splited_left_2[1]))
        # 处理y部分
        if splited_left_2[2] =='y':
            pass
        else:
            one_mr.append('eq2_y_index[{}]'.format(splited_left_2[2].replace('y','')))
    elif len(splited_left_2) == 4:
        # 有四个部分，看起来都要做填充
        # 第一个部分的符号
        one_mr.append('eq2_left_sym1[{}]'.format(splited_left_2[0]))
        if splited_left_2[1] =='x':
            pass
        else:
            one_mr.append('eq2_x_index[{}]'.format(splited_left_2[1].replace('x','')))
        # 第二个部分的符号
        one_mr.append('eq2_left_sym2[{}]'.format(splited_left_2[2]))
        # 处理y部分
        if splited_left_2[3] =='y':
            pass
        else:
            one_mr.append('eq2_y_index[{}]'.format(splited_left_2[3].replace('y','')))
    else:
        raise 'wrong input equation2, check preprocessing'

    # 处理第一个式子的右半部分
    if len(splited_right_2) == 1:
        # 只有一个实数，应该大部分都是如此
        one_mr.append('eq2_right_num1[{}]'.format(splited_right_2[0]))
    elif len(splited_right_2) == 3:
        # 有实数的计算存在，好吧，那就算一算吧
        one_mr.append('eq2_right_num1[{}]'.format(splited_right_2[0]))
        one_mr.append('eq2_right_sym[{}]'.format(splited_right_2[1]))
        one_mr.append('eq2_right_num2[{}]'.format(splited_right_2[2]))
    return ','.join(one_mr)




new_mr = []
for idx, row in train_set.iterrows():
    equ1, equ2 = row['关系1_trans'], row['关系2_trans']
    new_mr.append(get_equation_mr(equ1, equ2))
train_set['mr'] = new_mr
new_mr = []
for idx, row in dev_set.iterrows():
    equ1, equ2 = row['关系1_trans'], row['关系2_trans']
    new_mr.append(get_equation_mr(equ1, equ2))
dev_set['mr'] = new_mr
new_mr = []
for idx, row in test_set.iterrows():
    equ1, equ2 = row['关系1_trans'], row['关系2_trans']
    new_mr.append(get_equation_mr(equ1, equ2))
test_set['mr'] = new_mr


# In[6]:


def get_biparty(head, rel, tail,eqn='eq1'):
    one_dummy = 'dummy_cal'
    one_edge, one_node = [],[one_dummy]
    if rel == '+':
        one_edge += [(head, '和关系_{}'.format(eqn)), ('和关系_{}'.format(eqn), one_dummy),
                     (one_dummy, '和关系_{}_rev'.format(eqn)), ('和关系_{}_rev'.format(eqn), head),
                     (tail, '和关系_{}'.format(eqn)), ('和关系_{}_rev'.format(eqn), tail)]
        one_node += [head, tail, '和关系_{}'.format(eqn),'和关系_{}_rev'.format(eqn)]
    if rel == '-':
        one_edge += [(head,'被减数关系_{}'.format(eqn)), ('被减数关系_{}'.format(eqn), one_dummy),
                     (one_dummy,'被减数关系_{}_rev'.format(eqn)), ('被减数关系_{}_rev'.format(eqn), head),
                     (tail,'减数关系_{}'.format(eqn)), ('减数关系_{}'.format(eqn),one_dummy),
                    (one_dummy,'减数关系_{}_rev'.format(eqn)), ('减数关系_{}_rev'.format(eqn),tail)
                    ]
        one_node += [head, tail, '减数关系_{}'.format(eqn), '被减数关系_{}'.format(eqn),'减数关系_{}_rev'.format(eqn), '被减数关系_{}_rev'.format(eqn)]
    if rel == '*':
        one_edge += [(head,'被乘数关系_{}'.format(eqn)), ('被乘数关系_{}'.format(eqn), one_dummy),
                     (one_dummy,'被乘数关系_{}_rev'.format(eqn)), ('被乘数关系_{}_rev'.format(eqn), head),
                     (tail,'乘数关系_{}'.format(eqn)), ('乘数关系_{}'.format(eqn),one_dummy),
                    (one_dummy,'乘数关系_{}_rev'.format(eqn)), ('乘数关系_{}_rev'.format(eqn),tail)
                    ]
        one_node += [head, tail, '乘数关系_{}'.format(eqn), '被乘数关系_{}'.format(eqn), '乘数关系_{}_rev'.format(eqn), '被乘数关系_{}_rev'.format(eqn)]
    if rel == '/':
        one_edge += [(head,'被除数关系_{}'.format(eqn)), ('被除数关系_{}'.format(eqn), one_dummy),
                     (one_dummy,'被除数关系_{}_rev'.format(eqn)), ('被除数关系_{}_rev'.format(eqn), head),
                     (tail,'除数关系_{}'.format(eqn)), ('除数关系_{}'.format(eqn),one_dummy),
                    (one_dummy,'除数关系_{}_rev'.format(eqn)), ('除数关系_{}_rev'.format(eqn),tail)
                    ]
        one_node += [head, tail, '除数关系_{}'.format(eqn), '被除数关系_{}'.format(eqn),'被除数关系_{}_rev'.format(eqn),'除数关系_{}_rev'.format(eqn)]

    return one_edge, one_node, one_dummy




def bild_dual_new_graph(equ_1, equ_2, scene, var, mr_dict, tou_info, jiao_info):
    equ_1, equ_2 = equ_1.replace(' ',''), equ_2.replace(' ','')

    processed_var, processed_scene = process_var(var), process_scene(scene)
    x_ent, y_ent = mr_dict['x_ent'], mr_dict['y_ent']

    # equation graph
    all_node, all_edge = [x_ent, y_ent], []

    # common sense graph
    all_node_1, all_edge_1 = [x_ent, y_ent], []

    #加入场景信息
    for tail in re.split(r',|，',processed_scene[x_ent]):
        all_edge_1 += [(x_ent,'belong_to_x'),('belong_to_x',tail), (tail,'belong_to_x_rev'), ('belong_to_x_rev',x_ent)]
        all_node_1 += [tail,'belong_to_x','belong_to_x_rev']

    for tail in re.split(r',|，',processed_scene[y_ent]):
        all_edge_1 += [(y_ent,'belong_to_y'),('belong_to_y',tail), (tail,'belong_to_y_rev'), ('belong_to_y_rev',y_ent)]
        all_node_1 += [tail,'belong_to_y','belong_to_y_rev']

    # 处理头_entity, 头_unit, 脚_entity, 脚_unit
    if tou_info['entity'] != '':
        all_node_1 += [tou_info['entity'], '有头_ent','有头_ent_rev']
        all_edge_1 += [(x_ent,'有头_ent'), ('有头_ent', tou_info['entity']), (tou_info['entity'],'有头_ent_rev'),('有头_ent_rev', x_ent),
                     (y_ent, '有头_ent'), ('有头_ent_rev', y_ent)]
    if jiao_info['entity'] != '':
        all_node_1 += [jiao_info['entity'],'有脚_ent','有脚_ent_rev']
        all_edge_1 += [(x_ent, '有脚_ent'), ('有脚_ent', jiao_info['entity']), (jiao_info['entity'],'有脚_ent_rev'),('有脚_ent_rev', x_ent),
                     (y_ent, '有脚_ent'), ('有脚_ent_rev', y_ent)]
    if tou_info['unit'] != '':
        all_node_1 += [tou_info['unit'],'有头_unit','有头_unit_rev']
        all_edge_1 += [(x_ent,'有头_unit'),('有头_unit', tou_info['unit']), (tou_info['unit'],'有头_unit_rev'),('有头_unit_rev', x_ent),
                     (y_ent, '有头_unit'), ('有头_unit_rev', y_ent)]
    if jiao_info['unit'] != '':
        all_node_1 += [jiao_info['unit'],'有脚_unit','有脚_unit_rev']
        all_edge_1 += [(x_ent,'有脚_unit'),('有脚_unit', jiao_info['unit']), (jiao_info['unit'],'有脚_unit_rev'),('有脚_unit_rev', x_ent),
                     (y_ent, '有脚_unit'),('有脚_unit_rev', y_ent)]

    # 处理关系1左半部分
    tmp_x_eq1, tmp_y_eq1, tmp_x_eq2, tmp_y_eq2 = x_ent, y_ent, x_ent, y_ent
    tmp_right_eq1, tmp_right_eq2 = mr_dict['eq1_right_num1'], mr_dict['eq2_right_num1']
    if 'eq1_x_index' in mr_dict:
        all_edge += [(x_ent, '乘关系x'), ('乘关系x',mr_dict['eq1_x_index']), (mr_dict['eq1_x_index'],'乘关系x_rev'), ('乘关系x_rev',x_ent)]
        all_node += ['乘关系x', mr_dict['eq1_x_index'],'乘关系x_rev']
        tmp_x_eq1 = mr_dict['eq1_x_index']

    if 'eq1_y_index' in mr_dict:
        all_edge += [(y_ent,'乘关系y'), ('乘关系y', mr_dict['eq1_y_index']), (mr_dict['eq1_y_index'],'乘关系y_rev'), ('乘关系y_rev',y_ent)]
        all_node += ['乘关系y',mr_dict['eq1_y_index'], '乘关系y_rev']
        tmp_y_eq1 = mr_dict['eq1_y_index']

    # 处理关系2左半部分
    if 'eq2_x_index' in mr_dict:
        all_edge += [(x_ent, '乘关系x'), ('乘关系x',mr_dict['eq2_x_index']), (mr_dict['eq2_x_index'],'乘关系x_rev'), ('乘关系x_rev',x_ent)]
        all_node += ['乘关系x', mr_dict['eq2_x_index'], '乘关系x_rev']
        tmp_x_eq2 = mr_dict['eq2_x_index']

    if 'eq2_y_index' in mr_dict:
        all_edge += [(y_ent,'乘关系y'), ('乘关系y', mr_dict['eq2_y_index']), (mr_dict['eq2_y_index'],'乘关系y_rev'), ('乘关系y_rev',y_ent)]
        all_node += ['乘关系y',mr_dict['eq2_y_index'], '乘关系y_rev']
        tmp_y_eq2 = mr_dict['eq2_y_index']

    # 分析一下右边
    if 'eq1_right_sym' in mr_dict:
        # 右边存在一坨计算，需要算算
        one_tmp_edge1, one_tmp_node1, one_tmp_right1 = get_biparty(mr_dict['eq1_right_num1'], mr_dict['eq1_right_sym'],mr_dict['eq1_right_num2'],'eq1')
    else:
        one_tmp_edge1, one_tmp_node1, one_tmp_right1 = [], [mr_dict['eq1_right_num1']], mr_dict['eq1_right_num1']

    if 'eq2_right_sym' in mr_dict:
        one_tmp_edge2, one_tmp_node2, one_tmp_right2 = get_biparty(mr_dict['eq2_right_num1'], mr_dict['eq2_right_sym'],mr_dict['eq2_right_num2'],'eq2')
    else:
        one_tmp_edge2, one_tmp_node2, one_tmp_right2 = [], [mr_dict['eq2_right_num1']], mr_dict['eq2_right_num1']

    all_edge += one_tmp_edge1
    all_edge += one_tmp_edge2
    all_node += one_tmp_node1
    all_node += one_tmp_node2

    # 结合左边和右边
    if 'eq1_left_sym1' in mr_dict:
        # 这个时候只可能是eq1sym1=负数，eq1sym2=正数
        one_tmp_edge_inner_1, one_tmp_node_inner_1 = get_triparty(tmp_y_eq1, tmp_x_eq1, '-', one_tmp_right1,'eq1')
    else:
        one_tmp_edge_inner_1, one_tmp_node_inner_1 = get_triparty(tmp_x_eq1, tmp_y_eq1, mr_dict['eq1_left_sym2'], one_tmp_right1,'eq1')

    if 'eq2_left_sym1' in mr_dict:
        one_tmp_edge_inner_2, one_tmp_node_inner_2 = get_triparty(tmp_y_eq2, tmp_x_eq2, '-', one_tmp_right2,'eq2')
    else:
        one_tmp_edge_inner_2, one_tmp_node_inner_2 = get_triparty(tmp_x_eq2, tmp_y_eq2, mr_dict['eq2_left_sym2'], one_tmp_right2,'eq2')


    all_edge += one_tmp_edge_inner_1
    all_edge += one_tmp_edge_inner_2
    all_node += one_tmp_node_inner_1
    all_node += one_tmp_node_inner_2


    all_ele = [x for y in all_edge for x in y ]
    assert set(all_ele)==set(all_node), 'wrong preprocessing code'

    return list(all_edge), list(set(all_node)), list(all_edge_1), list(set(all_node_1))



def get_triparty(head_left_1, head_left_2, rel, right_one, eqn='eq1'):
    one_edge, one_node = [],[]
    if rel == '+':
        one_edge += [(head_left_1, '和关系_{}_res'.format(eqn)), ('和关系_{}_res'.format(eqn), right_one),
                     (right_one, '和关系_{}_rev_res'.format(eqn)), ('和关系_{}_rev_res'.format(eqn),head_left_1),
                     (head_left_2, '和关系_{}_res'.format(eqn)), ('和关系_{}_rev_res'.format(eqn),head_left_2),]
        one_node += [head_left_1, head_left_2, right_one, '和关系_{}_res'.format(eqn), '和关系_{}_rev_res'.format(eqn)]
    if rel == '-':
        one_edge += [(head_left_1,'被减数关系_{}_res'.format(eqn)), ('被减数关系_{}_res'.format(eqn), right_one),
                     (right_one,'被减数关系_{}_rev_res'.format(eqn)), ('被减数关系_{}_rev_res'.format(eqn), head_left_1),
                     (head_left_2,'减数关系_{}_res'.format(eqn)), ('减数关系_{}_res'.format(eqn),right_one),
                     (right_one,'减数关系_{}_rev_res'.format(eqn)), ('减数关系_{}_rev_res'.format(eqn),head_left_2)
                    ]
        one_node += [head_left_1, head_left_2, '减数关系_{}_res'.format(eqn), '被减数关系_{}_res'.format(eqn), right_one, '减数关系_{}_rev_res'.format(eqn), '被减数关系_{}_rev_res'.format(eqn)]
    if rel == '*':
        one_edge += [(head_left_1,'被乘数关系_{}_res'.format(eqn)), ('被乘数关系_{}_res'.format(eqn), right_one),
                     (right_one,'被乘数关系_{}_rev_res'.format(eqn)), ('被乘数关系_{}_rev_res'.format(eqn), head_left_1),
                     (head_left_2,'乘数关系_{}_res'.format(eqn)), ('乘数关系_{}_res'.format(eqn),right_one),
                    (right_one,'乘数关系_{}_rev_res'.format(eqn)), ('乘数关系_{}_rev_res'.format(eqn),head_left_2)
                    ]
        one_node += [head_left_1, head_left_2, '乘数关系_{}_res'.format(eqn), '被乘数关系_{}_res'.format(eqn),right_one,
                     '乘数关系_{}_rev_res'.format(eqn), '被乘数关系_{}_rev_res'.format(eqn)]
    if rel == '/':
        one_edge += [(head_left_1,'被除数关系_{}_res'.format(eqn)), ('被除数关系_{}_res'.format(eqn), right_one),
                     (right_one,'被除数关系_{}_rev_res'.format(eqn)), ('被除数关系_{}_rev_res'.format(eqn), head_left_1),
                     (head_left_2,'除数关系_{}_res'.format(eqn)), ('除数关系_{}_res'.format(eqn),right_one),
                    (right_one,'除数关系_{}_rev_res'.format(eqn)), ('除数关系_{}_rev_res'.format(eqn),head_left_2)
                    ]
        one_node += [head_left_1, head_left_2, '除数关系_{}_res'.format(eqn), '被除数关系_{}_res'.format(eqn),right_one,'被除数关系_{}_rev_res'.format(eqn), '除数关系_{}_rev_res'.format(eqn)]
    return one_edge, one_node


# In[9]:


class Constants:
    def __init__(self):
        self.BOS_WORD = '<s>'
        self.EOS_WORD = '</s>'
        self.PAD_WORD = '<blank>'
        self.UNK_WORD = '<unk>'
        self.eq1_x_index_WORD = 'eq_one_x_index'
        self.eq2_x_index_WORD = 'eq_two_x_index'
        self.eq1_y_index_WORD = 'eq_one_y_index'
        self.eq2_y_index_WORD = 'eq_two_y_index'
        self.eq1_right_num1_WORD = 'eq_one_right_num_one'
        self.eq2_right_num1_WORD = 'eq_two_right_num_one'
        self.eq1_right_num2_WORD = 'eq_one_right_num_two'
        self.eq2_right_num2_WORD = 'eq_two_right_num_two'
        self.x_entity_WORD = 'x_entity'
        self.y_entity_WORD = 'y_entity'
        self.head_info_unit_WORD = 'head_info_unit'
        self.jiao_info_unit_WORD = 'jiao_info_unit'
        self.jiao_info_entity_WORD = 'jiao_info_entity'
        self.head_info_entity_WORD = 'head_info_entity'
#         self.dummy_equal_WORD = 'dummy_cal'

        self.PAD = 0
        self.UNK = 1
        self.BOS = 2
        self.EOS = 3
        self.eq1_x_index = 4
        self.eq2_x_index = 5
        self.eq1_y_index = 6
        self.eq2_y_index = 7
        self.eq1_right_num1 = 8
        self.eq2_right_num1 = 9
        self.eq1_right_num2 = 10
        self.eq2_right_num2 = 11
        self.x_entity = 12
        self.y_entity = 13
        self.head_info_unit = 14
        self.jiao_info_unit = 15
        self.jiao_info_entity =16
        self.head_info_entity =17
#         self.dummy_equal = 18
Constants = Constants()



MR_FIELDS =['方程二右边数字一', '方程一右边数字一', 'x_entity','y_entity', 'head信息_entity',
    '脚信息_entity', 'head信息_unit', '脚信息_unit',
    '方程一y系数', '方程二y系数', '方程一x系数', '方程二x系数','方程一右边数字二','方程二右边数字二']

MR_KEYMAP = dict((key, idx) for idx, key in enumerate(MR_FIELDS))
MR_KEY_NUM = len(MR_FIELDS)

lex_fields = ['方程二右边数字一', '方程一右边数字一', 'x_entity', 'y_entity', 'head信息_entity', '脚信息_entity',
              'head信息_unit','脚信息_unit',
    '方程一y系数', '方程二y系数', '方程一x系数', '方程二x系数', '方程一右边数字二','方程二右边数字二']

lex_tar = [Constants.eq2_right_num1_WORD, Constants.eq1_right_num1_WORD, Constants.x_entity_WORD, Constants.y_entity_WORD, Constants.head_info_entity_WORD, Constants.jiao_info_entity_WORD,
    Constants.head_info_unit_WORD, Constants.jiao_info_unit_WORD,
    Constants.eq1_y_index_WORD,Constants.eq2_y_index_WORD,
    Constants.eq1_x_index_WORD, Constants.eq2_x_index_WORD, Constants.eq1_right_num2_WORD, Constants.eq2_right_num2_WORD]

lex_keymap = dict((key, idx) for idx, key in enumerate(lex_fields))


def process_jitu_mr_type(mr, scene, equ1, equ2, var, tou_info, jiao_info):
    mr, scene, equ1, equ2, var = copy.deepcopy(mr), copy.deepcopy(scene), copy.deepcopy(equ1), copy.deepcopy(equ2), copy.deepcopy(var)
    tou_info,jiao_info = eval(copy.deepcopy(tou_info)), eval(copy.deepcopy(jiao_info))
    lex_this = [None] * len(lex_tar)

    mr_dict = process_mr(mr, var)
#     print(mr_dict)
    ent_x, ent_y = mr_dict['x_ent'], mr_dict['y_ent']

    # 替换实体的名称作为输入
    if ent_y == ('超级'+ent_x):
        scene = scene.replace(ent_y, Constants.y_entity_WORD)
        mr_dict['y_ent'] = Constants.y_entity_WORD
        lex_this[lex_keymap['y_entity']] = ent_y
        scene = scene.replace(ent_x,Constants.x_entity_WORD)
        mr_dict['x_ent'] = Constants.x_entity_WORD
        lex_this[lex_keymap['x_entity']] = ent_x
    if ent_x == ('另'+ent_y):
        scene = scene.replace(ent_x, Constants.x_entity_WORD)
        mr_dict['x_ent'] = Constants.x_entity_WORD
        lex_this[lex_keymap['x_entity']] = ent_x
        scene = scene.replace(ent_y,Constants.y_entity_WORD)
        mr_dict['y_ent'] = Constants.y_entity_WORD
        lex_this[lex_keymap['y_entity']] = ent_y
    elif ent_y == ('另'+ent_x):
        scene = scene.replace(ent_y,Constants.y_entity_WORD)
        mr_dict['y_ent'] = Constants.y_entity_WORD
        lex_this[lex_keymap['y_entity']] = ent_y
        scene = scene.replace(ent_x, Constants.x_entity_WORD)
        mr_dict['x_ent'] = Constants.x_entity_WORD
        lex_this[lex_keymap['x_entity']] = ent_x
    elif ('另一' in ent_x) or ('另外一' in ent_y):
        scene = scene.replace(ent_y,Constants.y_entity_WORD)
        mr_dict['y_ent'] = Constants.y_entity_WORD
        lex_this[lex_keymap['y_entity']] = ent_y
        scene = scene.replace(ent_x, Constants.x_entity_WORD)
        mr_dict['x_ent'] = Constants.x_entity_WORD
        lex_this[lex_keymap['x_entity']] = ent_x
    elif ('另一' in ent_y) or ('另外一' in ent_x):
        scene = scene.replace(ent_x, Constants.x_entity_WORD)
        mr_dict['x_ent'] = Constants.x_entity_WORD
        lex_this[lex_keymap['x_entity']] = ent_x
        scene = scene.replace(ent_y,Constants.y_entity_WORD)
        mr_dict['y_ent'] = Constants.y_entity_WORD
        lex_this[lex_keymap['y_entity']] = ent_y
    else:
        if 'dummy' not in ent_x:
            scene = scene.replace(ent_x,Constants.x_entity_WORD)
            mr_dict['x_ent'] = Constants.x_entity_WORD
            lex_this[lex_keymap['x_entity']] = ent_x
            scene = scene.replace(ent_y,Constants.y_entity_WORD)
            mr_dict['y_ent'] = Constants.y_entity_WORD
            lex_this[lex_keymap['y_entity']] = ent_y

    items = mr.split(',')
    for idx, item in enumerate(items):
        key, raw_val = item.split('[')
#         if (key=='x_ent') & ('dummy' not in raw_val):
#             scene = scene.replace(ent_x,Constants.x_entity_WORD)
#             mr_dict['x_ent'] = Constants.x_entity_WORD
#             lex_this[lex_keymap['x_entity']] = ent_x
#         elif (key =='y_ent') & ('dummy' not in raw_val):
#             scene = scene.replace(ent_y,Constants.y_entity_WORD)
#             mr_dict['y_ent'] = Constants.y_entity_WORD
#             lex_this[lex_keymap['y_entity']] = ent_y
        if key=='eq1_x_index':
            mr_dict['eq1_x_index'] = Constants.eq1_x_index_WORD
            lex_this[lex_keymap['方程一x系数']] = raw_val[:-1]
        elif key == 'eq1_y_index':
            mr_dict['eq1_y_index'] = Constants.eq1_y_index_WORD
            lex_this[lex_keymap['方程一y系数']] = raw_val[:-1]
        elif key == 'eq1_left_sym2':
            pass
        elif key == 'eq1_right_num1':
            if mr_dict['eq1_right_num1'] == '0':
                mr_dict['eq1_right_num1'] = '0'
                lex_this[lex_keymap['方程一右边数字一']] = '0'
            else:
                mr_dict['eq1_right_num1'] = Constants.eq1_right_num1_WORD
                lex_this[lex_keymap['方程一右边数字一']] = raw_val[:-1]
        elif key == 'eq1_right_num2':
            mr_dict['eq1_right_num2'] = Constants.eq1_right_num2_WORD
            lex_this[lex_keymap['方程一右边数字二']] = raw_val[:-1]
        elif key == 'eq2_x_index':
            mr_dict['eq2_x_index'] = Constants.eq2_x_index_WORD
            lex_this[lex_keymap['方程二x系数']] = raw_val[:-1]
        elif key == 'eq2_y_index':
            mr_dict['eq2_y_index'] = Constants.eq2_y_index_WORD
            lex_this[lex_keymap['方程二y系数']] = raw_val[:-1]
        elif key == 'eq2_left_sym2':
            pass
        elif key == 'eq2_right_num1':
            if mr_dict['eq2_right_num1'] == '0':
                mr_dict['eq2_right_num1'] = '0'
                lex_this[lex_keymap['方程一右边数字一']] = '0'
            else:
                mr_dict['eq2_right_num1'] = Constants.eq2_right_num1_WORD
                lex_this[lex_keymap['方程二右边数字一']] = raw_val[:-1]
        elif key == 'eq2_right_num2':
            mr_dict['eq2_right_num2'] = Constants.eq2_right_num2_WORD
            lex_this[lex_keymap['方程二右边数字二']] = raw_val[:-1]
        else:
            pass
    # 一些特殊case，需要做一下处理，这里面是为了应对除法的情况
    if lex_this[lex_keymap['方程二右边数字一']] == lex_this[lex_keymap['方程一右边数字一']]:
        mr_dict['eq2_right_num1'] = Constants.eq1_right_num1_WORD
    # 特殊case 如果鸡兔互换
    if (lex_this[lex_keymap['方程一x系数']] == lex_this[lex_keymap['方程二y系数']]) & (lex_this[lex_keymap['方程二x系数']] == lex_this[lex_keymap['方程一y系数']]):
        mr_dict['eq2_y_index'] = Constants.eq1_x_index_WORD
        mr_dict['eq2_x_index'] = Constants.eq1_y_index_WORD

    if tou_info['entity'] != '':
        lex_this[lex_keymap['head信息_entity']] = tou_info['entity']
        tou_info['entity'] = Constants.head_info_entity_WORD
    if tou_info['unit'] != '':
        lex_this[lex_keymap['head信息_unit']] = tou_info['unit']
        tou_info['unit'] = Constants.head_info_unit_WORD
    if jiao_info['entity'] != '':
        lex_this[lex_keymap['脚信息_entity']] = jiao_info['entity']
        jiao_info['entity'] = Constants.jiao_info_entity_WORD
    if jiao_info['unit'] != '':
        lex_this[lex_keymap['脚信息_unit']] = jiao_info['unit']
        jiao_info['unit'] = Constants.jiao_info_unit_WORD
    all_edge, all_node, all_edge_1, all_node_1 = bild_dual_new_graph(equ1, equ2, scene, var, mr_dict, tou_info, jiao_info)
    return all_edge, all_node, lex_this, all_edge_1, all_node_1


edges, nodes, lexs, edges_2, nodes_2 = [],[],[],[],[]
for idx, row in train_set.iterrows():
    equ_1, equ_2, scene, var, mr, tou_info, jiao_info = row['关系1_trans'], row['关系2_trans'], row['scene'], row['变量'],row['mr'], row['头信息'],row['脚信息']
    edge_1, node_1, lex_1, edge_2, node_2 = process_jitu_mr_type(mr, scene, equ_1, equ_2, var, tou_info, jiao_info)
    edges.append(edge_1)
    nodes.append(node_1)
    lexs.append(lex_1)
    edges_2.append(edge_2)
    nodes_2.append(node_2)
train_set['edges'], train_set['nodes'], train_set['lexs'], train_set['edges_1'], train_set['nodes_1'] = edges, nodes, lexs, edges_2, nodes_2


edges, nodes, lexs, edges_2, nodes_2 = [],[],[],[],[]
for idx, row in dev_set.iterrows():
    equ_1, equ_2, scene, var, mr, tou_info, jiao_info = row['关系1_trans'], row['关系2_trans'], row['scene'], row['变量'],row['mr'], row['头信息'],row['脚信息']
    edge_1, node_1, lex_1, edge_2, node_2 = process_jitu_mr_type(mr, scene, equ_1, equ_2, var, tou_info, jiao_info)
    edges.append(edge_1)
    nodes.append(node_1)
    lexs.append(lex_1)
    edges_2.append(edge_2)
    nodes_2.append(node_2)
dev_set['edges'], dev_set['nodes'], dev_set['lexs'], dev_set['edges_1'], dev_set['nodes_1'] = edges, nodes, lexs, edges_2, nodes_2



edges, nodes, lexs, edges_2, nodes_2 = [],[],[],[],[]
for idx, row in test_set.iterrows():
    equ_1, equ_2, scene, var, mr, tou_info, jiao_info = row['关系1_trans'], row['关系2_trans'], row['scene'], row['变量'],row['mr'], row['头信息'],row['脚信息']
    edge_1, node_1, lex_1, edge_2, node_2 = process_jitu_mr_type(mr, scene, equ_1, equ_2, var, tou_info, jiao_info)
    edges.append(edge_1)
    nodes.append(node_1)
    lexs.append(lex_1)
    edges_2.append(edge_2)
    nodes_2.append(node_2)
test_set['edges'], test_set['nodes'], test_set['lexs'], test_set['edges_1'], test_set['nodes_1'] = edges, nodes, lexs, edges_2, nodes_2


def get_after_text(text, lex_list):
    if lex_list:
        for word, tar in zip(lex_list, lex_tar):
            if word is not None:
                if word in text:
                    text = text.replace(word, tar)
    return text
def tokenize_word(text, sp, lex_list=None):
    words = []
    if lex_list:
        for word, tar in zip(lex_list, lex_tar):
            if word is not None:
                if word in text:
                    text = text.replace(word, tar)
    for frag in sp.EncodeAsPieces(text):
        words.append(frag)
    return words


# 提取所有的场景信息
scene_new = train_set['scene'].tolist()
some_scene = set()
for one_scene in scene_new:
    one_scene = one_scene.replace(' ','')
    real_scene = re.findall(r'[(](.*?)[)]',one_scene)
    for sce in real_scene:
        cc_sce = re.split(',|，', sce)
        for cc in cc_sce:
            some_scene.add(cc)
all_scene = list(some_scene)


all_relation = [
    'belong_to_x','belong_to_y','belong_to_x_rev','belong_to_y_rev',
    '有头_ent','有头_ent_rev','有脚_ent','有脚_ent_rev',
    '有头_unit','有头_unit_rev','有脚_unit', '有脚_unit_rev','乘关系x','乘关系x_rev','乘关系y','乘关系y_rev',
    '和关系_eq1','和关系_eq2','被减数关系_eq1','被减数关系_eq2','减数关系_eq1','减数关系_eq2',
    '被乘数关系_eq1','被乘数关系_eq2','乘数关系_eq1','乘数关系_eq2','被除数关系_eq1','被除数关系_eq2',
    "除数关系_eq1","除数关系_eq2"
    '和关系_eq1_rev','和关系_eq2_rev','被减数关系_eq1_rev','被减数关系_eq2_rev','减数关系_eq1_rev','减数关系_eq2_rev',
    '被乘数关系_eq1_rev','被乘数关系_eq2_rev','乘数关系_eq1_rev','乘数关系_eq2_rev','被除数关系_eq1_rev','被除数关系_eq2_rev',
    "除数关系_eq1_rev","除数关系_eq2_rev",
    '减数关系_eq1_res','减数关系_eq1_rev_res','减数关系_eq2_res','减数关系_eq2_rev_res','和关系_eq1_res','和关系_eq1_rev_res',
    '和关系_eq2_res','和关系_eq2_rev_res','被减数关系_eq1_res','被减数关系_eq1_rev_res','被减数关系_eq2_res','被减数关系_eq2_rev_res'
]



after_sent = []
for idx, row in train_set.iterrows():
    text = row['ref']
    one_edge, one_node, lex_one = row['edges'], row['nodes'], row['lexs']
    after_sent.append(get_after_text(text, lex_one))
with open('./processed_data/after_text_with_rev_res.txt','w') as f:
    for line in after_sent:
        f.write(line+ '\n')
text_path = './processed_data/after_text_with_rev_res.txt'
model_save_path = './processed_data/encoded_with_rev_res'
user_symbols = ','.join(lex_tar+all_scene+all_relation+['dummy1', 'dummy2'])
spm.SentencePieceTrainer.Train('--input={}                             --model_prefix={} --model_type=bpe                             --user_defined_symbols={}                             --character_coverage=0.9996 --hard_vocab_limit=false'.format(
                                text_path, model_save_path, user_symbols))


sp = spm.SentencePieceProcessor()
sp.load('./processed_data/encoded_with_rev_res.model')





def build_vocab_idx(word_insts, min_word_count):
    '''Trim vocab by number of occurence'''
    full_vocab = set(w for sent in word_insts for w in sent)
    print('[Info] Original Vocabulary size =', len(full_vocab))
    word2idx = {
        Constants.BOS_WORD: Constants.BOS,
        Constants.EOS_WORD: Constants.EOS,
        Constants.PAD_WORD: Constants.PAD,
        Constants.UNK_WORD: Constants.UNK,
        Constants.eq1_x_index_WORD: Constants.eq1_x_index,
        Constants.eq2_x_index_WORD: Constants.eq2_x_index,
        Constants.eq1_y_index_WORD: Constants.eq1_y_index,
        Constants.eq2_y_index_WORD: Constants.eq2_y_index,
        Constants.eq1_right_num1_WORD: Constants.eq1_right_num1,
        Constants.eq2_right_num1_WORD: Constants.eq2_right_num1,
        Constants.eq1_right_num2_WORD: Constants.eq1_right_num2,
        Constants.eq2_right_num2_WORD: Constants.eq1_right_num2,
        Constants.x_entity_WORD: Constants.x_entity,
        Constants.y_entity_WORD: Constants.y_entity,
        Constants.head_info_unit_WORD: Constants.head_info_unit,
        Constants.jiao_info_unit_WORD: Constants.jiao_info_unit,
        Constants.jiao_info_entity_WORD: Constants.jiao_info_entity,
        Constants.head_info_entity_WORD: Constants.head_info_entity
    }

    word_count = {w: 0 for w in full_vocab}
    for sent in word_insts:
        for word in sent:
            word_count[word] += 1
    ignored_word_count = 0
    for word, count in word_count.items():
        if word not in word2idx:
            if count > min_word_count:
                word2idx[word] = len(word2idx)
            else:
                ignored_word_count+=1
    print('[Info] Trimmed vocabulary size = {},'.format(len(word2idx)), 'each with minimum occurence = {}'.format(min_word_count))
    print('[Info] Ingored word count = {}'.format(ignored_word_count))
    return word2idx


def get_scene(one_scene):
    one_scene = one_scene.replace(' ','')
    processed = process_scene(one_scene)
    all_word = set()
    for key, value in processed.items():
        for one_word in re.split(',|，', value):
            all_word.add(one_word)
    return list(all_word)



train_some_ref, train_some_scene,train_lex = [],[],[]
for idx, row in train_set.iterrows():
    edge, scene, node, ref, lex = row['edges'], row['scene'], row['nodes'], row['ref'], row['lexs']
    tokenized_sent = tokenize_word(ref, sp, lex)
    train_some_ref.append(tokenized_sent)
    train_some_scene.append(get_scene(scene))
    train_lex.append(lex)
dev_some_ref, dev_some_scene,dev_lex = [],[],[]
for idx, row in dev_set.iterrows():
    edge, scene, node, ref, lex = row['edges'], row['scene'], row['nodes'], row['ref'], row['lexs']
    tokenized_sent = tokenize_word(ref, sp, lex)
    dev_some_ref.append(tokenized_sent)
    dev_some_scene.append(get_scene(scene))
    dev_lex.append(lex)
test_some_ref, test_some_scene,test_lex = [],[],[]
for idx, row in test_set.iterrows():
    edge, scene, node, ref, lex = row['edges'], row['scene'], row['nodes'], row['ref'], row['lexs']
    tokenized_sent = tokenize_word(ref, sp, lex)
    test_some_ref.append(tokenized_sent)
    test_some_scene.append(get_scene(scene))
    test_lex.append(lex)



train_nodes_0= train_set['nodes'].tolist() #equation info node
train_nodes_1 = train_set['nodes_1'].tolist() #common sense info node
train_edges_0 = train_set['edges'].tolist() #equation info edge
train_edges_1 = train_set['edges_1'].tolist() #common sense info edge



word2idx = build_vocab_idx(train_nodes_0 + train_nodes_1 + train_some_ref, min_word_count=0)



data = {
    'dict':{
        'tgt': word2idx
    },
    "train":{
        'scene': train_some_scene,
        'ref': train_some_ref,
        'lexs': train_lex
    },
    "dev":{
        'scene': dev_some_scene,
        "ref": dev_some_ref,
        'lexs': dev_lex,
    },
    "test":{
        'scene': test_some_scene,
        "ref": test_some_ref,
        'lexs': test_lex
    }
}



# save一下dual graph的数据
data['train']['node_1'] = train_set['nodes'].tolist() #equation info node
data['train']['node_2'] = train_set['nodes_1'].tolist() #common sense info node
data['train']['edge_1'] = train_set['edges'].tolist() #equation info edge
data['train']['edge_2'] = train_set['edges_1'].tolist() #common sense info edge


data['dev']['node_1'] = dev_set['nodes'].tolist() #equation info node
data['dev']['node_2'] = dev_set['nodes_1'].tolist() #common sense info node
data['dev']['edge_1'] = dev_set['edges'].tolist() #equation info edge
data['dev']['edge_2'] = dev_set['edges_1'].tolist() #common sense info edge


data['test']['node_1'] = test_set['nodes'].tolist() #equation info node
data['test']['node_2'] = test_set['nodes_1'].tolist() #common sense info node
data['test']['edge_1'] = test_set['edges'] .tolist()#equation info edge
data['test']['edge_2'] = test_set['edges_1'].tolist() #common sense info edge


torch.save(data, './processed_data/dual_graph_rev_res.pt')

