from __future__ import annotations

from typing import Generator, TypeVar

from django.db.models.query import QuerySet

T = TypeVar("T")


class QuerysetIterator:
    def __init__(self, queryset: QuerySet[T], chunk_size=256, return_chunk=True):
        self.queryset = queryset
        self.chunk_size = chunk_size
        self.return_chunk = return_chunk

    def __len__(self):
        if self.return_chunk:
            return self.queryset.count() // self.chunk_size
        return self.queryset.count()

    def __iter__(self):
        return self

    def __next__(self):
        start = 0
        while True:
            chunk = self.queryset[start : start + self.chunk_size]

            if self.return_chunk:
                return chunk
            else:
                for obj in chunk:
                    return obj

            if len(chunk) < self.chunk_size:
                raise StopIteration()

            start += self.chunk_size
