#!/bin/bash

echo "=============================="
echo " Atualizando o repositório"
echo "=============================="

# Verifica o status atual
git status

# Adiciona todas as mudanças
git add .

# Pede mensagem de commit
echo "Digite a mensagem do commit:"
read mensagem

# Faz o commit
git commit -m "$mensagem"

# Detecta a branch atual automaticamente
BRANCH=$(git rev-parse --abbrev-ref HEAD)

# Faz o push
git push origin "$BRANCH"

echo "=============================="
echo " Push realizado para a branch '$BRANCH'"
echo "=============================="
