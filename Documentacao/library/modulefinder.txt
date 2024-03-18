"modulefinder" --- Procura módulos usados por um script
*******************************************************

**Código-fonte:** Lib/modulefinder.py

======================================================================

Este módulo fornece uma classe "ModuleFinder" que pode ser usada para
determinar o conjunto de módulos importados por um script. O
"modulefinder.py" também pode ser executado como um script, fornecendo
o nome do arquivo de um script Python como seu argumento, após o qual
um relatório dos módulos importados será impresso.

modulefinder.AddPackagePath(pkg_name, path)

   Registra que o pacote chamado *pkg_name* pode ser encontrado no
   caminho especificado em *path*

modulefinder.ReplacePackage(oldname, newname)

   Permite especificar que o módulo chamado *oldname* é de fato o
   pacote chamado *newname*.

class modulefinder.ModuleFinder(path=None, debug=0, excludes=[], replace_paths=[])

   Esta classe fornece os métodos "run_script()" e "report()" para
   determinar o conjunto de módulos importados por um script. *path*
   pode ser uma lista de diretórios para procurar por módulos; se não
   especificado, "sys.path" é usado. *debug* define o nível de
   depuração; valores mais altos fazem a classe imprimir mensagens de
   depuração sobre o que está fazendo. *excludes* é uma lista de nomes
   de módulos a serem excluídos da análise. *replace_paths* é uma
   lista de tuplas "(oldpath, newpath)" que serão substituídas nos
   caminhos dos módulos.

   report()

      Imprime um relatório na saída padrão que lista os módulos
      importados pelo script e seus caminhos, bem como os módulos que
      estão faltando ou parecem estar ausentes.

   run_script(pathname)

      Analisa o conteúdo do arquivo *pathname*, que deve conter o
      código Python.

   modules

      Um nome de módulo de mapeamento de dicionário para módulos. Veja
      Exemplo de uso de ModuleFinder.


Exemplo de uso de "ModuleFinder"
================================

O script que será analisado posteriormente (bacon.py):

   import re, itertools

   try:
       import baconhameggs
   except ImportError:
       pass

   try:
       import guido.python.ham
   except ImportError:
       pass

O script que irá gerar o relatório de bacon.py:

   from modulefinder import ModuleFinder

   finder = ModuleFinder()
   finder.run_script('bacon.py')

   print('Loaded modules:')
   for name, mod in finder.modules.items():
       print('%s: ' % name, end='')
       print(','.join(list(mod.globalnames.keys())[:3]))

   print('-'*50)
   print('Modules not imported:')
   print('\n'.join(finder.badmodules.keys()))

Exemplo de saída (pode variar dependendo da arquitetura):

   Loaded modules:
   _types:
   copyreg:  _inverted_registry,_slotnames,__all__
   re._compiler:  isstring,_sre,_optimize_unicode
   _sre:
   re._constants:  REPEAT_ONE,makedict,AT_END_LINE
   sys:
   re:  __module__,finditer,_expand
   itertools:
   __main__:  re,itertools,baconhameggs
   re._parser:  _PATTERNENDERS,SRE_FLAG_UNICODE
   array:
   types:  __module__,IntType,TypeType
   ---------------------------------------------------
   Modules not imported:
   guido.python.ham
   baconhameggs
