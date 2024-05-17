from __future__ import print_function
import time
import url2io_client
from url2io_client.rest import ApiException
from pprint import pprint

# Configure API key authorization: token_in_query
configuration = url2io_client.Configuration()
configuration.host = 'http://url2api.applinzi.com' # 你申请的服务地址，默认为体验版地址：http://url2api.applinzi.com
configuration.api_key['token'] = 'oI0c0LkpS_qj2WacpAsQjA'
# Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
# configuration.api_key_prefix['token'] = 'Bearer'

# create an instance of the API class
api_instance = url2io_client.URL2ArticleApi(url2io_client.ApiClient(configuration))
url = 'https://new.qq.com/rain/a/20240517A049V900' # str | 要提取正文网页的网址，参考 [URL Encoding](http://www.w3schools.com/tags/ref_urlencode.asp)
fields=['text'] # list[str] | 指示需要额外返回的额外字段，取值为：  - `next`: 表示要提取下一页链接。   - `text`: 表示要返回正文的纯文字格式。   - `markdown`: 表示返回正文的markdown格式。   构造url时多个值通过','号隔开，如`?fields=text,next`。调用sdk时使用列表即可，如fields= ['text', 'markdown']。 (optional)
param_callback = 'param_callback_example' # str | 使用jsonp实现Ajax跨域请求时需要传此参数 (optional)

try:
    # 网页结构智能解析 HTTP Get 接口
    api_response = api_instance.get_article(url, fields=fields)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling URL2ArticleApi->get_article: %s\n" % e)
