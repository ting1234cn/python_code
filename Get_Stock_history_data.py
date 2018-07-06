import pandas as pd
import tushare as ts
import time
year=2017
quater=4
stock_list=["600196","600460","600276","603993","600298","002258"]
start_date="2016-07-01"
current_date=time.strftime("%F")
writer = pd.ExcelWriter(r"stocks.xlsx")
for stock in stock_list :
    if int(stock)>600000 :
        df=ts.sh_margin_details(start=start_date, end=current_date,symbol=stock)


    df_k=ts.get_hist_data(start=start_date, end=current_date,code=stock)
    if df_k is None:
        break
    if df is None:
        df_k.to_excel(writer, stock, index=False)
    else:
        df_merge=pd.merge(df,df_k,left_on='opDate',right_index=True)
        df_merge.drop(['stockCode','securityAbbr','rqmcl','rqyl','rqchl','price_change','p_change'],axis=1,inplace=True)
        df_merge.to_excel(writer, stock, index=False)



writer.save()