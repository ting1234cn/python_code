import time

import pandas as pd
import tushare as ts
from WindPy import w

year=2017
quater=4
stock_list=["600196","600460","600276","603993","600298","600177","002258","002507","002415","601318","000725","601318","002415"]
start_date="2016-06-30"

current_hour=time.strftime("%H", time.localtime())
if int(current_hour)>15 :   #下午4点以后才能去当天的数据
    current_date=time.strftime("%F")
else:                        #否则取前一天的日期
    current_date=time.strftime("%Y-%m-%d", time.localtime(time.time( )-86400))



def get_stock_history_tu():
    writer = pd.ExcelWriter(r"stocks.xlsx")
    for stock in stock_list :
        if int(stock)>600000 :
            df=ts.sh_margin_details(start=start_date, end=current_date,symbol=stock)
        else:
            break
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

def get_wind_data(wdata):
    i=0
    df = pd.DataFrame()
    df["Date"]=wdata.Times
    for field in wdata.Fields:
        df[field] = wdata.Data[i]
        i = i + 1

    return df


def get_stock_history_wind():
    writer = pd.ExcelWriter(r"stocks.xlsx")
    w.start();

    if w.isconnected():
        for stock in stock_list :
            if int(stock) > 600000:
                stock_code=stock+".SH"
                wdata = w.wsd(stock_code,
                              "high,open,low,close,volume,mrg_long_bal,mrg_short_vol_bal,mfd_netbuyamt,pe_ttm,pb_lf,BOLL,MACD,roe_avg",
                              start_date, current_date,
                              "unit=1;traderType=1;BOLL_N=26;BOLL_Width=2;BOLL_IO=1;MACD_L=26;MACD_S=12;MACD_N=9;MACD_IO=1;MA_N=7;Fill=Previous;PriceAdj=F")
            else:
                stock_code=stock+".SZ"
                wdata = w.wsd(stock_code, "high,open,low,close,volume,volume,mfd_netbuyamt,mfd_netbuyamt_a,pe_ttm,BOLL,MACD,yoyeps_basic,yoy_equity,mfd_buyvol_m",
                          start_date, current_date,
                              "unit=1;traderType=1;BOLL_N=26;BOLL_Width=2;BOLL_IO=1;MACD_L=26;MACD_S=12;MACD_N=9;MACD_IO=1;MA_N=7;Fill=Previous;PriceAdj=F")


            df_k=get_wind_data(wdata)
            if df_k is None:
                break
            df_k.plot()



            df_k.to_csv(stock+".txt",sep="\t",index=False)
            df_k.to_excel(writer, stock, index=False)
        writer.save()
    else:
        print("wind is not connected!!")
        return


if __name__ == '__main__':
    get_stock_history_wind()