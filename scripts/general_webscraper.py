from functools import wraps
from time import time
import pandas as pd
import urllib2
import re


'''decorator to record time'''
def timed(f):
  @wraps(f)
  def wrapper(*args, **kwds):
    start = time()
    result = f(*args, **kwds)
    elapsed = time() - start
    print "%s took %d time to finish" % (f.__name__, elapsed)
    return result
  return wrapper    

@timed
def stack_df(series,function,show=0,force=0):
    '''
    key function
    takes a function that acts on one url and collects all the results in a df of applying that function to
    each url in a series
    '''
    df = pd.DataFrame()
    for lk in series:
        if show == 1:
            print lk
        if force == 0:
            temp = function(lk)
            df = df.append(temp,ignore_index=True)
        elif force == 1:
            try:
                temp = function(lk)
                df = df.append(temp,ignore_index=True)
            except:
                print lk, ' skipped due to errors'                
    return df


#could refactor
def download(url):
    
    #need to understand why this works
    hdr = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
       'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
       'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
       'Accept-Encoding': 'none',
       'Accept-Language': 'en-US,en;q=0.8',
       'Connection': 'keep-alive'}

    req = urllib2.Request(url, headers=hdr)
    
    try:
        response = urllib2.urlopen(req)
    except urllib2.HTTPError, e:
        print('error')
        
    page_source = response.read()
    
    return page_source

def cleaner(text):
    #removes non breaking line spaces from a string
    if re.search(u'\xa0',text):
        return text.replace(u'\xa0',' ')
    else:
        return text