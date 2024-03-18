Exceções
********

**Código-fonte:** Lib/asyncio/exceptions.py

======================================================================

exception asyncio.TimeoutError

   Um apelido descontinuado de "TimeoutError", levantado quando a
   operação excedeu o prazo determinado.

   Alterado na versão 3.11: Esta classe foi feita como um apelido de
   "TimeoutError".

exception asyncio.CancelledError

   A operação foi cancelada.

   Esta exceção pode ser capturada para executar operações
   personalizadas quando as tarefas assíncronas são canceladas. Em
   quase todas as situações, a exceção deve ser levantada novamente.

   Alterado na versão 3.8: "CancelledError" é agora uma subclasse de
   "BaseException" em vez de "Exception".

exception asyncio.InvalidStateError

   Estado interno inválido de "Task" ou "Future".

   Pode ser levantada em situações como definir um valor de resultado
   para um objeto *Future* que já tem um valor de resultado definido.

exception asyncio.SendfileNotAvailableError

   A *syscall* "sendfile" não está disponível para o soquete ou tipo
   de arquivo fornecido.

   Uma subclasse de "RuntimeError".

exception asyncio.IncompleteReadError

   A operação de leitura solicitada não foi totalmente concluída.

   Levantada pelas APIs de fluxo de asyncio.

   Esta exceção é uma subclasse de "EOFError".

   expected

      O número total ("int") de bytes esperados.

   partial

      Uma string de "bytes" lida antes do final do fluxo ser
      alcançado.

exception asyncio.LimitOverrunError

   Atingiu o limite de tamanho do buffer ao procurar um separador.

   Levantada pelas APIs de fluxo de asyncio.

   consumed

      O número total de bytes a serem consumidos.
