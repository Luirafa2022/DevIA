"quopri" --- Codifica e decodifica dados MIME imprimidos entre aspas
********************************************************************

**Código-fonte:** Lib/quopri.py

======================================================================

Este módulo realiza codificação e decodificação de transporte
imprimida entre aspas, como definido em **RFC 1521**: "MIME
(Multipurpose Internet Mail Extensions) Part One: Mechanisms for
Specifying and Describing the Format of Internet Message Bodies". A
codificação imprimida entre aspas é projetada para dados em que há
relativamente poucos caracteres não imprimíveis; o esquema de
codificação base64 disponível através do módulo "base64" é mais
compacto se existirem muitos desses caracteres, como no envio de um
arquivo gráfico.

quopri.decode(input, output, header=False)

   Decodifica o conteúdo do arquivo *input* e escreve os dados
   binários decodificados resultantes no arquivo *output*. *input* e
   *output* devem ser *objetos arquivos binários*. Se o argumento
   opcional *header* estiver presente e for true, o sublinhado será
   decodificado como espaço. Isso é usado para decodificar cabeçalhos
   codificados em "Q", conforme descrito em **RFC 1522**: "MIME
   (Multipurpose Internet Mail Extensions) Part Two: Message Header
   Extensions for Non-ASCII Text".

quopri.encode(input, output, quotetabs, header=False)

   Codifica o conteúdo do arquivo *input* e grava os dados imprimíveis
   entre aspas resultantes no arquivo *output*. *input* e *output*
   devem ser *objetos arquivos binários*. *quotetabs*, um sinalizador
   não opcional que controla a codificação de espaços e tabulações
   incorporados; quando true, codifica esses espaços em branco
   incorporados e, quando false, os deixa sem codificação. Observe que
   os espaços e tabulações que aparecem no final das linhas são sempre
   codificados, conforme **RFC 1521**. *header* é um sinalizador que
   controla se os espaços são codificados como sublinhados, conforme
   **RFC 1522**.

quopri.decodestring(s, header=False)

   Como "decode()", exceto pelo fato de aceitar uma fonte "bytes" e
   retornar o correspondente decodificado "bytes".

quopri.encodestring(s, quotetabs=False, header=False)

   Como "encode()", exceto pelo fato de aceitar uma fonte "bytes" e
   retornar o "bytes" codificado correspondente. Por padrão, envia um
   valor "False" para o parâmetro *quotetabs* da função "encode()".

Ver também:

  Módulo "base64"
     Codifica e decodifica dados de base64 MIME
