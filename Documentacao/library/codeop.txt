"codeop" --- Compila código Python
**********************************

**Código-fonte:** Lib/codeop.py

======================================================================

O módulo "codeop" fornece utilitários sobre os quais o loop de
leitura-execução-exibição do Python pode ser emulado, como é feito no
módulo "code". Como resultado, você provavelmente não deseja usar o
módulo diretamente; se você deseja incluir tal loop em seu programa,
você provavelmente deseja usar o módulo "code".

Há duas partes para esta tarefa:

1. Ser capaz de dizer se uma linha de entrada completa uma instrução
   Python: em suma, dizer se deve exibir '">>>"' ou '"..."' em
   seguida.

2. Lembrar quais instruções futuras o usuário inseriu, para que as
   entradas subsequentes possam ser compiladas com essas declarações
   em vigor.

O módulo "codeop" fornece uma maneira de fazer cada uma dessas coisas
e uma maneira de fazer as duas coisas.

Para fazer apenas a primeira:

codeop.compile_command(source, filename='<input>', symbol='single')

   Tenta compilar *source*, que deve ser uma string de código Python e
   retornar um objeto código se *source* for um código Python válido.
   Nesse caso, o atributo de nome de arquivo do objeto código será
   *filename*, cujo padrão é "'<input>'". Retorna "None" se *source*
   *não* é um código Python válido, mas é um prefixo de código Python
   válido.

   Se houver um problema com *source*, uma exceção será levantada.
   "SyntaxError" é levantada se houver sintaxe Python inválida, e
   "OverflowError" ou "ValueError" se houver um literal inválido.

   O argumento *symbol* determina se *source* é compilado como uma
   instrução ("'single'", o padrão), como uma sequência de
   *instruções* ("'exec'") ou como uma *expressão* ("'eval'").
   Qualquer outro valor fará com que "ValueError" seja levantada.

   Nota:

     É possível (mas não provável) que o analisador sintático pare de
     analisar com um resultado bem-sucedido antes de chegar ao final
     da fonte; neste caso, os símbolos finais podem ser ignorados em
     vez de causar um erro. Por exemplo, uma barra invertida seguida
     por duas novas linhas pode ser seguida por lixo arbitrário. Isso
     será corrigido quando a API para o analisador for melhor.

class codeop.Compile

   Instâncias desta classe têm métodos "__call__()" idênticos em
   assinatura à função embutida "compile()", mas com a diferença de
   que se a instância compilar o texto do programa contendo uma
   instrução "__future__", a instância se "lembra" e compila todos os
   textos de programa subsequentes com a instrução em vigor.

class codeop.CommandCompiler

   Instâncias desta classe têm métodos "__call__()" idênticos em
   assinatura a "compile_command()"; a diferença é que se a instância
   compila o texto do programa contendo uma instrução "__future__", a
   instância se "lembra" e compila todos os textos do programa
   subsequentes com a instrução em vigor.
