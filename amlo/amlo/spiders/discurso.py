import scrapy

class DiscursoSpider(scrapy.Spider):

    name = "discurso"

    start_urls = ['https://lopezobrador.org.mx/transcripciones/']

    # Scrapy parser for link and conference title

    def parse(self, response):

        for entry in response.xpath('//article'):

            entry_dict = {}

            # Parse link of conference content

            entry_dict['link'] = entry.xpath('.//a/@href').get()

            # Parse title of conference content

            entry_dict['title'] = entry.xpath('.//a/@title').get()

            # Open conference/entry content

            yield scrapy.Request(entry_dict['link'],
                                meta={'meta_item':entry_dict}, callback=self.parse_entry)

        # Get next page

        next_page = response.css('a[class*=next]::attr(href)').get()

        if next_page:
            yield response.follow(next_page, callback=self.parse)

    # Scrapy parser for the conference transcript

    def parse_entry(self, response):

        entry_dict = response.meta['meta_item']

        paragraphs = response.xpath('//div[@class="entry-content"]/p').getall()

        '''
        BeautifulSoup(paragraphs[4], "html.parser").find('p').contents
        '''

        entry_dict['text'] = "|".join(paragraphs)

        yield entry_dict
