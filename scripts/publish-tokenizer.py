"""Rebuild the bilingual note and discovery surfaces from checked-in Markdown.
Requires beautifulsoup4, Node.js, markdown-it and katex (resolved through NODE_PATH).
"""
from pathlib import Path
from bs4 import BeautifulSoup as Soup
import subprocess, xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
SLUG='claude-16k-tokenizer-revisited'
OLD='claude-16k-tokenizer-reasoning-architecture'
BASE='https://syhdsgwater.github.io'
DATA={
 'zh':('为什么 Claude 选择了 16K 词表？','一个漂亮假说的失败：两组实验、更多 token，以及性能与计费之间的关系。','zh-CN','2026 年 9 月 13 日','约 25 分钟'),
 'en':('Why did Claude choose a 16K vocabulary tokenizer?','An elegant hypothesis fails: two experiments, more tokens, and the connection between performance and billing.','en','13 September 2026','25 min read')}

css='''
.article-body img {display:block;width:100%;height:auto;margin:1.8rem 0}
.article-body table {width:100%;border-collapse:collapse;font-size:.88em;line-height:1.65;margin:1.8rem 0}
.article-body td,.article-body th {padding:12px 14px;border-bottom:1px solid #dedbd3;vertical-align:top;text-align:left;overflow-wrap:anywhere}
.article-body th {background:#f7f6f2;font-weight:600}
.article-body .table-scroll {overflow-x:auto;margin:1.5rem 0}
.article-body .table-scroll table {min-width:560px;margin:0}
.article-body .katex-display {overflow-x:auto;overflow-y:hidden;padding:8px 0;max-width:100%}
.article-body .article-callout {border-left:2px solid #c46743;padding:0 0 0 22px;margin-bottom:2.5rem;background:none}
.article-body p {overflow-wrap:anywhere}
.article-body h2 {scroll-margin-top:30px}
.article-toc ol {padding-left:1.2rem}
.article-toc a {line-height:1.55}
@media(max-width:640px){.article-body table{font-size:14px}.article-body td,.article-body th{padding:10px}.article-body .katex{font-size:.95em}}
'''

for lang,(title,deck,locale,date,readtime) in DATA.items():
 source=ROOT/'content'/SLUG/f'{lang}.md'
 rendered=source.with_suffix('.html')
 subprocess.run(['node',str(ROOT/'scripts/render-tokenizer.cjs'),str(source),str(rendered)],check=True)
 body=Soup(rendered.read_text(encoding='utf8'),'html.parser');rendered.unlink()
 soup=Soup((ROOT/lang/'notes'/OLD/'index.html').read_text(encoding='utf8'),'html.parser')
 soup.html['lang']=locale
 for old_note in soup.select('.article-margin-note'):old_note.decompose()
 soup.title.string=title+' · JiangHongwei'
 for tag in soup.select('meta[name="description"],meta[property="og:description"]'):tag['content']=deck
 for tag in soup.select('meta[property="og:title"]'):tag['content']=title
 for tag in soup.select('meta[property="article:published_time"]'):tag['content']='2026-09-13T00:00:00.000Z'
 for tag in soup.select('link[rel="canonical"]'):tag['href']=f'{BASE}/{lang}/notes/{SLUG}/'
 for tag in soup.select('link[hreflang]'):
  target='zh' if tag['hreflang'].startswith('zh') else 'en';tag['href']=f'{BASE}/{target}/notes/{SLUG}/'
 for tag in soup.select('meta[property="og:url"]'):tag['content']=f'{BASE}/{lang}/notes/{SLUG}/'
 for tag in soup.select('.language-switch a'):
  target='zh' if tag.get_text(strip=True) in ['中','中文'] else 'en';tag['href']=f'/{target}/notes/{SLUG}/'
 hero=soup.select_one('.article-hero')
 hero.select_one('h1').string=title
 hero.select_one('.article-deck').string=deck
 spans=hero.select('.article-meta-line span');spans[1].string=date;spans[2].string=readtime
 hero.select_one('.article-author').find_all('span')[-1].string=('发布于 ' if lang=='zh' else 'Published ')+date
 for element in hero.select('.article-comparison'):element.decompose()
 article=soup.select_one('.article-body');article.clear()
 headings=[]
 for i,h in enumerate(body.select('h2,h3')):
  h['id']=f'section-{i+1}';headings.append((h.name,h['id'],h.get_text()))
 for table in body.select('table'):
  first=table.find('tr')
  for td in first.find_all('td',recursive=False):td.name='th';td['scope']='col'
  wrapper=body.new_tag('div',attrs={'class':'table-scroll','tabindex':'0','role':'region','aria-label':'实验数据表' if lang=='zh' else 'Experiment data table'})
  table.wrap(wrapper)
 for img in body.select('img'):
  img['alt']='Anthropic 2026 ARR：公开口径与 tokenizer 归一化情景' if lang=='zh' else 'Anthropic 2026 ARR: reported revenue and tokenizer-normalized scenario'
  img['width']='1600';img['height']='860';img['loading']='lazy'
 for element in list(body.contents):article.append(element)
 toc=soup.select_one('.article-toc ol');toc.clear()
 for level,ident,text in headings:
  if level!='h2':continue
  li=soup.new_tag('li');a=soup.new_tag('a',href='#'+ident);a.string=text;li.append(a);toc.append(li)
 footer=soup.select_one('.article-footer');footer.find('p').string=title
 for a in footer.select('a[href*="/notes/"]'):
  target='en' if lang=='zh' else 'zh';a['href']=f'/{target}/notes/{SLUG}/';a.string='English' if target=='en' else '中文'
 style=soup.new_tag('style');style.string=css;soup.head.append(style)
 link=soup.new_tag('link',rel='stylesheet',href='/styles/katex/katex.min.css');soup.head.append(link)
 dest=ROOT/lang/'notes'/SLUG/'index.html';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(str(soup),encoding='utf8')

home=Soup((ROOT/'index.html').read_text(encoding='utf8'),'html.parser')
listing=home.select_one('.notes-list')
for li in listing.find_all('li',recursive=False):
 if li.select_one(f'a[href*="{SLUG}"]'):li.decompose()
li=Soup(f'''<li><div class="notes-list-meta"><time datetime="2026-09-13">13 Sep 2026</time><span>Tokenizer · experiments · economics</span></div><div class="notes-list-copy"><h3>Why Claude chose 16K</h3><p>Two experiments challenge the bottleneck hypothesis. More tokens connect performance with billing.</p></div><div class="notes-list-actions"><a href="/en/notes/{SLUG}/" lang="en">English</a><a href="/zh/notes/{SLUG}/" lang="zh-CN">中文</a></div></li>''','html.parser').li
listing.insert(0,li)
latest=home.select_one('.latest-note')
latest.select('.note-index span')[1].string='13 Sep 2026'
latest.select_one('h2').string='Why Claude chose 16K'
latest.select_one('.latest-deck').string=DATA['en'][1]
for a,lang in zip(latest.select('.read-actions a'),['en','zh']):a['href']=f'/{lang}/notes/{SLUG}/'
figure=latest.select_one('figure');figure.clear();figure['class']='tokenizer-arr-preview'
img=home.new_tag('img',src='/images/claude-tokenizer/anthropic-arr-tokenizer-2026.en.svg',alt='Anthropic ARR and tokenizer-normalized scenario',width='1600',height='860',style='width:100%;height:auto;display:block;')
figure.append(img)
for item in latest.select('.benchmark-contrast'):item.decompose()
(ROOT/'index.html').write_text(str(home),encoding='utf8')

tree=ET.parse(ROOT/'rss.xml');channel=tree.getroot().find('channel')
for item in list(channel.findall('item')):
 if SLUG in item.findtext('link',''):channel.remove(item)
for lang in ['zh','en']:
 item=ET.Element('item');title,deck,locale,*_=DATA[lang]
 for key,value in {'title':title,'link':f'{BASE}/{lang}/notes/{SLUG}/','guid':f'{BASE}/{lang}/notes/{SLUG}/','pubDate':'Sun, 13 Sep 2026 00:00:00 GMT','language':locale,'description':deck}.items():ET.SubElement(item,key).text=value
 channel.insert(4,item)
ET.indent(tree);tree.write(ROOT/'rss.xml',encoding='utf-8',xml_declaration=True)
ns='http://www.sitemaps.org/schemas/sitemap/0.9';ET.register_namespace('',ns)
tree=ET.parse(ROOT/'sitemap.xml');root=tree.getroot()
for entry in list(root):
 if SLUG in entry.findtext(f'{{{ns}}}loc',''):root.remove(entry)
for lang in ['en','zh']:
 entry=ET.SubElement(root,f'{{{ns}}}url')
 for k,v in {'loc':f'{BASE}/{lang}/notes/{SLUG}/','lastmod':'2026-09-13T00:00:00.000Z','changefreq':'monthly','priority':'0.9'}.items():ET.SubElement(entry,f'{{{ns}}}{k}').text=v
root[0].find(f'{{{ns}}}lastmod').text='2026-09-13T00:00:00.000Z'
ET.indent(tree);tree.write(ROOT/'sitemap.xml',encoding='utf-8',xml_declaration=True)
print('Built both editions, homepage, RSS and sitemap.')
