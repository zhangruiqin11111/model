# coding=gbk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
# ����׼��
tele = pd.read_csv('tele_camp_ok.csv')
# print(tele.describe(include='all'))

all_X = tele.columns[4:]

# '''��ȡ������������������Ա����ʹ��'''
continuous_X = ['PromCnt12', 'PromCnt36', 'PromCntMsg12',
                'PromCntMsg36', 'Age', 'AvgARPU', 'AvgHomeValue', 'AvgIncome']
categorical_X = list(set(all_X) - set(continuous_X))
print(categorical_X,continuous_X)

# ���ַ����������б���
from sklearn.preprocessing import LabelEncoder
print(tele.Gender.head(),tele.HomeOwner.head())
le = LabelEncoder()
tele['Gender'] = le.fit_transform(tele['Gender'])    # ��Gender�����Զ�����
tele['HomeOwner'].replace({'H': 0, 'U': 1}, inplace=True)
print(tele.Gender.head(),tele.HomeOwner.head())
# �鿴����ȱʧ��
print(tele.info())
# ��ARPU�Ƿ�ȱʧֵ��Ϊ������
arpu_known = tele[tele['ARPU'].notnull()].iloc[:, :].copy()
arpu_unknown = tele[tele['ARPU'].isnull()].iloc[:, :].copy()
print(arpu_known.ARPU,arpu_unknown)
# ��ȱʧֵARPU��������Է���
arpu_known.plot('ARPU', 'AvgARPU', kind='scatter')
plt.show()
# ����Ƥ��ѷ���ϵ��
print(arpu_known.corr(method='pearson'))


# �����Իع�
lm_s = ols('ARPU ~ AvgARPU', data=arpu_known).fit()
print(lm_s.summary())


# ��Զ���Իع�
formula = 'ARPU ~' + '+'.join(continuous_X)
lm = ols(formula, data=arpu_known).fit()
print(lm.summary())

# ���Իع����� �в�
ana1 = lm_s
arpu_known['Pred'] = ana1.predict(arpu_known)
arpu_known['resid'] = ana1.resid
print(ana1.resid)
arpu_known.plot('AvgARPU', 'resid',kind='scatter')
plt.show()
# ����ǿӰ���
# ��׼��
resid_mean = arpu_known['resid'].mean(); resid_std = arpu_known['resid'].std()

arpu_known['resid_t'] = (arpu_known['resid'] - resid_mean) / resid_std
print(arpu_known[abs(arpu_known['resid_t']) > 2].head())

# ɾ��ǿӰ���
arpu_known2 = arpu_known[abs(arpu_known['resid_t']) <= 2].copy()
ana2 = ols('ARPU ~ AvgARPU', arpu_known2).fit()
arpu_known2['Pred'] = ana2.predict(arpu_known2)
arpu_known2['resid'] = ana2.resid
arpu_known2.plot('AvgARPU', 'resid', kind='scatter')
plt.show()
print(ana2.rsquared)

# # statemodels���ṩ�˸���ǿӰ����ж�ָ��
from statsmodels.stats.outliers_influence import OLSInfluence

print(OLSInfluence(ana1).summary_frame().head())

# ��Ԫ���Իع�ı���ɸѡ
# ������ǰѡ�񷨱���ɸѡ����
def forward_select(data, response):
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = float('inf'), float('inf')
    while remaining:
        aic_with_candidates=[]
        for candidate in remaining:
            formula = "{} ~ {}".format(response,' + '.join(selected + [candidate]))
            aic = ols(formula=formula, data=data).fit().aic
            aic_with_candidates.append((aic, candidate))
        aic_with_candidates.sort(reverse=True)
        best_new_score, best_candidate=aic_with_candidates.pop()
        if current_score > best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
            print ('aic is {},continuing!'.format(current_score))
        else:
            print ('forward selection over!')
            break
    formula = "{} ~ {} ".format(response,' + '.join(selected))
    print('final formula is {}'.format(formula))
    model = ols(formula=formula, data=data).fit()
    return(selected, model)

candidate_var = list(continuous_X)
candidate_var.append('ARPU')
data_for_select = arpu_known2[candidate_var]

selected_var, lm_m = forward_select(data=data_for_select, response='ARPU')  #  ǰ��ѡ�����
print(lm_m.rsquared)

# ���ع��������� ���ع����Իᵼ�½ؾ� �ع�ϵ�������ȶ� ����������������Ϻͼ�����ع����Զ����Իع��Ӱ��
def vif(df, col_i):
    cols = list(df.columns)
    cols.remove(col_i)
    cols_noti = cols
    formula = col_i + '~' + '+'.join(cols_noti)
    r2 = ols(formula, df).fit().rsquared
    return 1. / (1. - r2)

exog = arpu_known2[selected_var]

for i in exog.columns:
    print(i, '\t', vif(df=exog, col_i=i))

# ���ӷ������
selected_var.extend(['C(Class)', 'C(Gender)'])
formula2 = 'ARPU ~' + '+'.join(selected_var)
ana3 = ols(formula2, arpu_known2).fit()
print(ana3.summary())

# ���ӽ���ЧӦ
selected_var.append('C(Class):C(Gender)')
formula3 = 'ARPU ~' + '+'.join(selected_var)
ana4 = ols(formula3, arpu_known2).fit()
print(ana4.summary())
#
selected_var.remove('C(Gender)')
selected_var.remove('C(Class):C(Gender)')
selected_var.remove('Age')

formula4 = 'ARPU ~' + '+'.join(selected_var)
ana5 = ols(formula4, arpu_known2).fit()
