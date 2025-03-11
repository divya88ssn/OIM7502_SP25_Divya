import scrapy
#scrapy crawl sp500 -o ../../../data/sp500_data.csv

class Sp500Spider(scrapy.Spider):
    name = 'sp500'
    allowed_domains = ['slickcharts.com']
    start_urls = ['https://www.slickcharts.com/sp500/performance']

    def parse(self, response):
        rows = response.xpath('//table[contains(@class, "table")]/tbody/tr')

        for row in rows:
            number = row.xpath('./td[1]/text()').get()
            company = row.xpath('./td[2]/a/text()').get()
            symbol = row.xpath('./td[3]/a/text()').get()

            # The YTD return text node comes after the <img> inside td[4]
            ytd_return_raw = row.xpath('./td[4]//text()').getall()
            # Join and strip extra whitespace
            ytd_return = ''.join([text.strip() for text in ytd_return_raw if text.strip()]).replace('&nbsp;', '')

            yield {
                'Number': number.strip() if number else '',
                'Company': company.strip() if company else '',
                'Symbol': symbol.strip() if symbol else '',
                'YTD Return': ytd_return
            }
