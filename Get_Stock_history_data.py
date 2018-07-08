import pandas as pd
import tushare as ts
from WindPy import w
from datetime import *
import numpy as np
import time
year=2017
quater=4
stock_list=["600196","600460","600276","603993","600298","002258"]
start_date="2016-07-01"
current_date=time.strftime("%F")
def get_stock_hisory_tu():
    writer = pd.ExcelWriter(r"stocks.xlsx")
    for stock in stock_list :
        if int(stock)>600000 :
            df=ts.sh_margin_details(start=start_date, end=current_date,symbol=stock)


        df_k=ts.get_hist_data(start=start_date, end=current_date,code=stock)
        if df_k is None:
            break
        if int(stock)<600000:
            df_k.to_excel(writer, stock, index=True)
        else:
            df_merge=pd.merge(df,df_k,left_on='opDate',right_index=True)
            df_merge.drop(['stockCode','securityAbbr','rqmcl','rqyl','rqchl','price_change','p_change'],axis=1,inplace=True)
            df_merge.sort_values(by='opDate',inplace=True)
            df_merge.to_excel(writer, stock, index=False)
    writer.save()

def get_stock_history_wind():
    writer = pd.ExcelWriter(r"stocks.xlsx")
    w.start();
    if w.isconnected():
        for stock in stock_list :
            if int(stock) > 600000:
                wdata=w.wsd("002258.SZ", "open,high,low,close,volume,mrg_long_bal", start_date, current_date, "unit=1;Fill=Previous;PriceAdj=F")
                df_k=pd.DataFrame()
                i=0
                for field in wdata.Fields :
                    df_k[field]=wdata.Data[i]
                    i=i+1

            wdata=w.wsd("002258.SZ", "mrg_long_bal,mfd_netbuyamt,mfd_inflowrate_m,roe_yearly,ocftoor,yoyeps_basic,yoyprofit", "2017-07-01", "2018-07-06", "unit=1;Fill=Previous;traderType=1;PriceAdj=F")
            df = pd.DataFrame()
            i = 0
            for field in wdata.Fields:
                df[field] = wdata.Data[i]
                i = i + 1
            if df_k is None:
                break


            df_merge=pd.merge(df,df_k,left_on='opDate',right_index=True)
            df_merge.drop(['stockCode','securityAbbr','rqmcl','rqyl','rqchl','price_change','p_change'],axis=1,inplace=True)
            df_merge.sort_values(by='opDate',inplace=True)
            df_merge.to_excel(writer, stock, index=False)
        writer.save()
    else:
        print("wind is not connected!!")
        return


if __name__ == '__main__':
    get_stock_history_wind()