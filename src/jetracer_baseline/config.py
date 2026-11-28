# -*- coding: utf-8 -*-
"""Nap config YAML (co bo doc/ghi du phong thuan Python khi Jetson chua cai PyYAML).

Python 3.6 compatible (Jetson Nano / JetPack 4.5.1). Khong dung dataclasses,
khong dung f-string, khong dung type union `X | Y`.
"""

import ast
import copy
import io
import os
import re

try:
    import yaml as _pyyaml
except ImportError:
    _pyyaml = None

_HEX_INT_RE = re.compile(r'^[+-]?0[xX][0-9a-fA-F]+$')
_DEC_INT_RE = re.compile(r'^[+-]?[0-9]+$')
_FLOAT_RE = re.compile(r'^[+-]?(?:[0-9]+\.[0-9]*|\.[0-9]+|[0-9]+)(?:[eE][+-]?[0-9]+)?$')


def _strip_yaml_comment(line):
    """Cat comment `# ...` ben ngoai chuoi nhay don / nhay kep."""
    in_s = False
    in_d = False
    esc = False
    for i, ch in enumerate(line):
        if esc:
            esc = False
            continue
        if ch == '\\' and in_d:
            esc = True
            continue
        if ch == "'" and not in_d:
            in_s = not in_s
        elif ch == '"' and not in_s:
            in_d = not in_d
        elif ch == '#' and not in_s and not in_d:
            if i == 0 or line[i - 1] in (' ', '\t'):
                return line[:i]
    return line


def _split_top_level(text, sep=','):
    """Tach theo dau `sep` o cap ngoac 0 (ho tro list/dict long nhau)."""
    parts = []
    cur = []
    depth = 0
    in_s = False
    in_d = False
    esc = False
    for ch in text:
        if esc:
            cur.append(ch)
            esc = False
            continue
        if ch == '\\' and in_d:
            cur.append(ch)
            esc = True
            continue
        if ch == "'" and not in_d:
            in_s = not in_s
            cur.append(ch)
        elif ch == '"' and not in_s:
            in_d = not in_d
            cur.append(ch)
        elif not in_s and not in_d:
            if ch in ('[', '{'):
                depth += 1
                cur.append(ch)
            elif ch in (']', '}'):
                depth -= 1
                cur.append(ch)
            elif ch == sep and depth == 0:
                parts.append(''.join(cur).strip())
                cur = []
            else:
                cur.append(ch)
        else:
            cur.append(ch)
    tail = ''.join(cur).strip()
    if tail:
        parts.append(tail)
    return parts


def _parse_scalar(val):
    s = val.strip()
    if not s or s in ('null', 'Null', 'NULL', '~'):
        return None
    if s in ('true', 'True', 'TRUE'):
        return True
    if s in ('false', 'False', 'FALSE'):
        return False
    if (len(s) >= 2) and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        try:
            return ast.literal_eval(s)
        except Exception:
            return s[1:-1]
    if s.startswith('[') and s.endswith(']'):
        inner = s[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(item) for item in _split_top_level(inner, ',')]
    if s.startswith('{') and s.endswith('}'):
        inner = s[1:-1].strip()
        if not inner:
            return {}
        out = {}
        for pair in _split_top_level(inner, ','):
            if ':' in pair:
                k, v = pair.split(':', 1)
                out[_parse_scalar(k)] = _parse_scalar(v)
        return out
    if _HEX_INT_RE.match(s):
        return int(s, 16)
    if _DEC_INT_RE.match(s):
        return int(s)
    if _FLOAT_RE.match(s):
        try:
            return float(s)
        except ValueError:
            pass
    return s


def _find_key_colon(text):
    """Tim dau `:` phan tach key: value ben ngoai ngoac nhay."""
    in_s = False
    in_d = False
    for i, ch in enumerate(text):
        if ch == "'" and not in_d:
            in_s = not in_s
        elif ch == '"' and not in_s:
            in_d = not in_d
        elif ch == ':' and not in_s and not in_d:
            if i + 1 == len(text) or text[i + 1] in (' ', '\t'):
                return i
    return -1


def _fallback_yaml_load(stream_or_text):
    """Bo doc YAML thuan Python cho moi file trong `configs/*.yaml`."""
    text = stream_or_text.read() if hasattr(stream_or_text, 'read') else str(stream_or_text)
    tokens = []
    for raw in text.splitlines():
        clean = _strip_yaml_comment(raw).rstrip()
        if not clean.strip():
            continue
        indent = len(clean) - len(clean.lstrip(' '))
        tokens.append((indent, clean.lstrip(' ')))
    if not tokens:
        return None

    def _parse_block(idx, base_indent):
        if idx >= len(tokens):
            return None, idx
        if tokens[idx][1].startswith('- ') or tokens[idx][1] == '-':
            items = []
            while idx < len(tokens):
                ind, txt = tokens[idx]
                if ind < base_indent or not (txt.startswith('- ') or txt == '-'):
                    break
                if ind > base_indent:
                    break
                rest = txt[2:].strip() if txt.startswith('- ') else ''
                idx += 1
                if not rest:
                    if idx < len(tokens) and tokens[idx][0] > ind:
                        sub, idx = _parse_block(idx, tokens[idx][0])
                        items.append(sub)
                    else:
                        items.append(None)
                else:
                    cpos = _find_key_colon(rest)
                    if cpos > 0 and not rest.startswith(('"', "'", '[', '{')):
                        k = _parse_scalar(rest[:cpos])
                        v_txt = rest[cpos + 1:].strip()
                        item = {}
                        if v_txt:
                            item[k] = _parse_scalar(v_txt)
                        elif idx < len(tokens) and tokens[idx][0] > ind:
                            item[k], idx = _parse_block(idx, tokens[idx][0])
                        else:
                            item[k] = None
                        if idx < len(tokens) and tokens[idx][0] > ind:
                            more, idx = _parse_block(idx, tokens[idx][0])
                            if isinstance(more, dict):
                                item.update(more)
                        items.append(item)
                    else:
                        items.append(_parse_scalar(rest))
            return items, idx

        mapping = {}
        while idx < len(tokens):
            ind, txt = tokens[idx]
            if ind < base_indent:
                break
            if ind > base_indent:
                break
            if txt.startswith('- ') or txt == '-':
                break
            cpos = _find_key_colon(txt)
            if cpos < 0:
                idx += 1
                continue
            key = _parse_scalar(txt[:cpos])
            val_txt = txt[cpos + 1:].strip()
            idx += 1
            if val_txt:
                mapping[key] = _parse_scalar(val_txt)
            else:
                if idx < len(tokens) and (
                        tokens[idx][0] > ind
                        or (tokens[idx][0] == ind
                            and (tokens[idx][1].startswith('- ')
                                 or tokens[idx][1] == '-'))):
                    sub, idx = _parse_block(idx, tokens[idx][0])
                    mapping[key] = sub
                else:
                    mapping[key] = None
        return mapping, idx

    result, _ = _parse_block(0, tokens[0][0])
    return result


def _fmt_yaml_scalar(val):
    if val is None:
        return 'null'
    if isinstance(val, bool):
        return 'true' if val else 'false'
    if isinstance(val, (int, float)):
        return repr(val)
    if isinstance(val, list):
        return '[' + ', '.join(_fmt_yaml_scalar(x) for x in val) + ']'
    if isinstance(val, dict):
        return '{' + ', '.join(
            '%s: %s' % (_fmt_yaml_scalar(k), _fmt_yaml_scalar(val[k]))
            for k in sorted(val.keys())) + '}'
    s = str(val)
    if (not s or any(ch in s for ch in (':', '#', '[', ']', '{', '}', ',', '\n'))
            or s in ('true', 'false', 'null', 'True', 'False', 'None')):
        return "'%s'" % s.replace("'", "''")
    return s


def _fallback_yaml_dump(data, indent=0):
    lines = []
    pad = ' ' * indent
    if isinstance(data, dict):
        for k in sorted(data.keys()):
            v = data[k]
            if isinstance(v, dict) and v:
                lines.append('%s%s:' % (pad, k))
                lines.append(_fallback_yaml_dump(v, indent + 2).rstrip('\n'))
            elif isinstance(v, list) and v and any(isinstance(x, dict) for x in v):
                lines.append('%s%s:' % (pad, k))
                for item in v:
                    lines.append('%s  - %s' % (pad, _fmt_yaml_scalar(item)))
            else:
                lines.append('%s%s: %s' % (pad, k, _fmt_yaml_scalar(v)))
    else:
        lines.append('%s%s' % (pad, _fmt_yaml_scalar(data)))
    return '\n'.join(lines) + '\n'


def yaml_safe_load(stream_or_text):
    """Doc YAML bang PyYAML neu co, tu dong dung bo doc thuan Python neu chua cai."""
    if _pyyaml is not None:
        return _pyyaml.safe_load(stream_or_text)
    return _fallback_yaml_load(stream_or_text)


def yaml_safe_dump(data, default_flow_style=False, allow_unicode=True):
    """Ghi YAML bang PyYAML neu co, tu dong dung bo ghi thuan Python neu chua cai."""
    if _pyyaml is not None:
        return _pyyaml.safe_dump(
            data, default_flow_style=default_flow_style, allow_unicode=allow_unicode)
    return _fallback_yaml_dump(data)


class Config(object):
    """Truy cap config kieu `cfg.get('control.v_max', 0.2)` hoac `cfg['control']`."""

    def __init__(self, data):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]

    def get(self, dotted_key, default=None):
        node = self._data
        for part in dotted_key.split('.'):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, dotted_key, value):
        parts = dotted_key.split('.')
        node = self._data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def as_dict(self):
        return copy.deepcopy(self._data)


def _deep_merge(base, override):
    """Merge `override` vao `base` theo chieu sau, tra ve dict moi."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path, overrides=None):
    """Nap `path`, roi ap lan luot cac file trong `overrides` (list duong dan)."""
    if not os.path.exists(path):
        raise IOError('Khong tim thay config: ' + path)
    # encoding tuong minh: config co comment tieng Viet, Windows mac dinh cp1252
    with io.open(path, 'r', encoding='utf-8') as fh:
        data = yaml_safe_load(fh) or {}

    for extra in (overrides or []):
        if not os.path.exists(extra):
            raise IOError('Khong tim thay override config: ' + extra)
        with io.open(extra, 'r', encoding='utf-8') as fh:
            data = _deep_merge(data, yaml_safe_load(fh) or {})

    return Config(data)
