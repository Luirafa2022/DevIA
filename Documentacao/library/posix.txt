"posix" --- As chamadas de sistema mais comuns do POSIX
*******************************************************

======================================================================

Este módulo fornece acesso à funcionalidade do sistema operacional
padronizada pelo padrão C e pelo padrão POSIX (uma interface Unix
levemente disfarçada).

Disponibilidade: Unix.

**Não importe este módulo diretamente.** Em vez disso, importe o
módulo "os", que fornece uma versão *portátil* dessa interface. No
Unix, o módulo "os" fornece um superconjunto da interface "posix". Em
sistemas operacionais não Unix, o módulo "posix" não está disponível,
mas um subconjunto está sempre disponível na interface "os". Uma vez
que "os" é importado, seu uso *não* causa penalidade de desempenho em
comparação com "posix". Além disso, "os" fornece algumas
funcionalidades adicionais, como chamar automaticamente "putenv()"
quando uma entrada em "os.environ" é alterada.

Erros são relatados como exceções. As exceções usuais são dadas para
erros de tipo, enquanto os erros relatados pelas chamadas do sistema
levantam "OSError".


Suporte a arquivos grandes
==========================

Vários sistemas operacionais (incluindo AIX e Solaris) fornecem
suporte a arquivos maiores que 2 GiB a partir de um modelo de
programação C em que int e long são valores de 32 bits. Isso
geralmente é realizado definindo o tamanho relevante e os tipos de
deslocamento como valores de 64 bits. Esses arquivos às vezes são
chamados de *arquivos grandes*.

O suporte a arquivos grandes é ativado no Python quando o tamanho de
um "off_t" é maior que a long e long long é pelo menos tão grande
quanto um "off_t". Pode ser necessário configurar e compilar o Python
com certos sinalizadores do compilador para ativar esse modo. Por
exemplo, com o Solaris 2.6 e 2.7, você precisa fazer algo como:

   CFLAGS="`getconf LFS_CFLAGS`" OPT="-g -O2 $CFLAGS" \
           ./configure

Em sistemas Linux com capacidade para arquivos grandes, isso pode
funcionar:

   CFLAGS='-D_LARGEFILE64_SOURCE -D_FILE_OFFSET_BITS=64' OPT="-g -O2 $CFLAGS" \
           ./configure


Conteúdo notável do módulo
==========================

Além de muitas funções descritas na documentação do módulo "os",
"posix" define o seguinte item de dados:

posix.environ

   Um dicionário que representa o ambiente de strings no momento em
   que o interpretador foi iniciado. Chaves e valores são bytes no
   Unix e str no Windows. Por exemplo, "environ[b'HOME']"
   ("environ['HOME']" no Windows) é o nome do caminho do diretório
   inicial, equivalente a "getenv("HOME")" em C .

   A modificação deste dicionário não afeta o ambiente de strings
   passado por "execv()", "popen()" ou "system()". Se você precisar
   alterar o ambiente, passe "environ" para "execve()" ou adicione
   atribuições de variável e instruções de exportação para a string de
   comando para "system()" ou "popen()".

   Alterado na versão 3.2: No Unix, chaves e valores são bytes.

   Nota:

     O módulo "os" fornece uma implementação alternativa de "environ"
     que atualiza o ambiente ao ocorrerem modificações. Observe também
     que a atualização de "os.environ" tornará este dicionário
     obsoleto. O uso do módulo "os" é recomendado sobre o acesso
     direto ao módulo "posix".
