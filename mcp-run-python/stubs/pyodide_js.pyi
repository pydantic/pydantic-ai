class _ToPy:
    def to_py(self) -> dict[str, str]:
        ...


class _Api:
    _import_name_to_package_name: _ToPy

_api: _Api
