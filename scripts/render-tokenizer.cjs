// npm install --prefix <tools-dir> markdown-it katex; set NODE_PATH=<tools-dir>/node_modules
const fs = require('fs');
const md = require('markdown-it')({html:true, typographer:false});
const katex = require('katex');
const input=fs.readFileSync(process.argv[2],'utf8');
let math=[];
let s=input.replace('**TL;DR: **','**TL;DR:** ').replace(/\$\$([\s\S]*?)\$\$|\$`([\s\S]*?)`\$/g, (_,display,inline)=> {
 const i=math.length;
 math.push(katex.renderToString(display ?? inline,{displayMode:display!==undefined,throwOnError:true,output:'htmlAndMathml'}));
 return `MATHPLACEHOLDER${i}END`;
});
s=s.replace(/<table_of_contents\/>|<empty-block\/>/g,'').replace(/<mention-page[^>]*\/>/g,'[Claude 的 16K 赌注](/zh/notes/claude-16k-tokenizer-reasoning-architecture/)');
s=s.replace(/<callout[^>]*>([\s\S]*?)<\/callout>/g,(_,body)=>'\n<div class="article-callout">\n\n'+body.replace(/^\t+/gm,'').replace(/\n(!\[)/g,'\n\n$1')+'\n\n</div>\n');
s=s.replace(/<table[^>]*>([\s\S]*?)<\/table>/g,(_,body)=>'<table>'+body.replace(/<td>([\s\S]*?)<\/td>/g,(_,cell)=>'<td>'+md.renderInline(cell)+'</td>')+'</table>\n\n');
// Notion separates paragraphs with a single newline. Preserve every paragraph.
let lines=s.split('\n'), out=[];
for(let i=0;i<lines.length;i++) {
 let line=lines[i];
 if(/^#{1,2} /.test(line)) line='#'+line;
 out.push(line);
 if(line.trim() && !/^\s*[<|*-]/.test(line) && !/^\s*[<|*-]/.test(lines[i+1]||'') && lines[i+1]?.trim()) out.push('');
}
let html=md.render(out.join('\n')).replace(/MATHPLACEHOLDER(\d+)END/g,(_,i)=>math[+i]);
fs.writeFileSync(process.argv[3],html);
