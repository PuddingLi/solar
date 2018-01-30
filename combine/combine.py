
from sklearn.externals import joblib
import pandas as pd
import numpy as np

def finderrorcode(brand, info):
    file = brand + '.m'
    clf = joblib.load(file)
    y_predict = clf.predict([info])
    return y_predict

def distance(x, y):
    d1=np.sqrt(np.sum(np.square(x-y)))
    return d1

def distToCentrolid(Vector, KMeansModel):

  cluster = KMeansModel.predict([Vector])
  #得到该聚类中心的质心
  t =  cluster[0]
  centrolid = KMeansModel.cluster_centers_[t,:]
  #计算距离
  d = distance(centrolid, Vector)
  return t,d

def isabnormal(brand, info):

    file = brand + '.pkl'
    clf = joblib.load(file)

    file2 = brand + '_threshold.csv'
    final = pd.read_csv(file2)
    a, b = distToCentrolid(info, clf)

    if b < final.at[a, 'b']:
        result = finderrorcode(brand, info)
        str = 'There will be a failure and the code is '+ result
    else:
        str = 'There will not be a failure'
    return str

def run():
    #brand = 'F058CD9F-89D7-4A1B-BB48-94EFA500AB95'
    brand = '2E4DFD95-6467-4B5A-A460-847A17353BF5'
    info = [4,1,0,0,1,0.0,348.0,31.0,0.0,0.0,0.0,0.0,87.18799999999999,0,0,0.0,269.2,0.0,0.0,575.0,0.0,0.0,272.6,321.5,49.91,0,0,272.9,321.5,0,0,273.4,321.5,0,0,0.0,152.2,0.0,1.0]

    label = ['DEVICE_TYPE_ID', 'DEVICE_ONLINE_FLAG'
        , 'DEVICE_FLAG', 'DEVICE_STOP', 'STATUS', 'Battery_tem', 'Irr', 'Sur_tem', 'Windspeed', 'humidity', 'Pressure',
             'rainfall'
        , 'winddirection', 'Soil_tem', 'Soil_humidity', 'totalrd', 'Ipv', 'Ipvpv1', 'Ipv2', 'Vpv'
        , 'Vpvpv1', 'Vpv2', 'rvac', 'rIac', 'rFac', 'rPac', 'rZac', 'svac', 'sIac', 'sPac', 'sZac', 'tvac',
             'tIac', 'tPac', 'tZac', 'itemp', 'Pac','ReactivePv','PvFactor']

    str = isabnormal(brand, info)
    print(str)

if __name__ == '__main__':
    run()

