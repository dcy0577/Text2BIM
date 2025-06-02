#!/usr/bin/env python
# coding=utf-8

# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
import difflib
from collections.abc import Mapping
import sys
import traceback
from typing import Any, Callable, Dict

class InterpretorError(ValueError):
    """
    An error raised when the interpretor cannot evaluate a Python expression, due to syntax error or unsupported
    operations.
    """
    pass

class PersistentInterpreter:
    """
    A persistent Python interpreter that maintains state between executions
    and supports extended Python syntax.
    """
    
    def __init__(self, tools: Dict[str, Callable] = None):
        self.global_state = {}
        self.tools = tools or {}
        self.local_state_stack = []  # For nested scopes
        
    def execute(self, code: str, chat_mode=False):
        """Execute code while maintaining persistent state"""
        return evaluate(code, self.tools, self.global_state, chat_mode)

def evaluate(code: str, tools: Dict[str, Callable], state=None, chat_mode=False):
    """
    Enhanced evaluate function with better error handling and state management
    """
    try:
        expression = ast.parse(code)
    except SyntaxError as e:
        print(f"Syntax error in generated code: {e}")
        return None, state
    
    if state is None:
        state = {}
    
    result = None
    code_lines = code.split('\n')
    
    for idx, node in enumerate(expression.body):
        try:
            line_result = evaluate_ast(node, state, tools)
        except InterpretorError as e:
            error_line = getattr(node, 'lineno', idx + 1)
            error_content = code_lines[error_line - 1] if error_line <= len(code_lines) else '<unknown>'
            msg = f"Evaluation stopped at line {error_line}: {e}\nLine content: {error_content}"
            print(msg)
            raise InterpretorError(msg) from None
        except Exception as e:
            error_line = getattr(node, 'lineno', idx + 1)
            error_content = code_lines[error_line - 1] if error_line <= len(code_lines) else '<unknown>'
            msg = f"Unexpected error at line {error_line}: {e}\nLine content: {error_content}"
            print(msg)
            raise Exception(msg) from None
            
        if line_result is not None:
            result = line_result
    
    return result, state

def evaluate_ast(expression: ast.AST, state: Dict[str, Any], tools: Dict[str, Callable]):
    """
    Enhanced AST evaluator with support for more Python syntax
    """
    
    # Existing supported nodes (from original code)
    if isinstance(expression, ast.Assign):
        return evaluate_assign(expression, state, tools)
    elif isinstance(expression, ast.Call):
        return evaluate_call(expression, state, tools)
    elif isinstance(expression, ast.Constant):
        return expression.value
    elif isinstance(expression, ast.Dict):
        keys = [evaluate_ast(k, state, tools) for k in expression.keys]
        values = [evaluate_ast(v, state, tools) for v in expression.values]
        return dict(zip(keys, values))
    elif isinstance(expression, ast.Expr):
        return evaluate_ast(expression.value, state, tools)
    elif isinstance(expression, ast.For):
        return evaluate_for(expression, state, tools)
    elif isinstance(expression, ast.FormattedValue):
        return evaluate_ast(expression.value, state, tools)
    elif isinstance(expression, ast.If):
        return evaluate_if(expression, state, tools)
    elif isinstance(expression, ast.JoinedStr):
        return "".join([str(evaluate_ast(v, state, tools)) for v in expression.values])
    elif isinstance(expression, ast.List):
        return [evaluate_ast(elt, state, tools) for elt in expression.elts]
    elif isinstance(expression, ast.Name):
        return evaluate_name(expression, state, tools)
    elif isinstance(expression, ast.Subscript):
        return evaluate_subscript(expression, state, tools)
    elif isinstance(expression, ast.BinOp):
        return evaluate_binop(expression, state, tools)
    elif isinstance(expression, ast.Tuple):
        return tuple(evaluate_ast(element, state, tools) for element in expression.elts)
    elif isinstance(expression, ast.Import):
        return evaluate_import(expression, state)
    elif isinstance(expression, ast.ImportFrom):
        return evaluate_import_from(expression, state)
    elif isinstance(expression, ast.Attribute):
        return evaluate_attribute(expression, state, tools)
    elif isinstance(expression, ast.AugAssign):
        return evaluate_aug_assign(expression, state, tools)
    elif isinstance(expression, ast.IfExp):
        return evaluate_ifexp(expression, state, tools)
    elif isinstance(expression, ast.UnaryOp):
        return evaluate_unaryop(expression, state, tools)
    elif isinstance(expression, ast.ListComp):
        return evaluate_listcomp(expression, state, tools)
    elif isinstance(expression, ast.FunctionDef):
        return evaluate_function_def(expression, state, tools)
    elif isinstance(expression, ast.Starred):
        return evaluate_ast(expression.value, state, tools)
    elif isinstance(expression, ast.Compare):
        return evaluate_compare(expression, state, tools)
    elif isinstance(expression, ast.BoolOp):
        return evaluate_boolop(expression, state, tools)
    elif isinstance(expression, ast.Return):
        return evaluate_return(expression, state, tools)
    elif isinstance(expression, ast.Pass):
        return None
    elif isinstance(expression, ast.Raise):
        return evaluate_raise(expression, state, tools)
    elif isinstance(expression, ast.Assert):
        return evaluate_assert(expression, state, tools)
    elif isinstance(expression, ast.Try):
        return evaluate_try(expression, state, tools)
    
    elif isinstance(expression, ast.While):
        return evaluate_while(expression, state, tools)
    elif isinstance(expression, ast.Break):
        raise BreakException()
    elif isinstance(expression, ast.Continue):
        raise ContinueException()
    elif isinstance(expression, ast.ClassDef):
        return evaluate_class_def(expression, state, tools)
    elif isinstance(expression, ast.Lambda):
        return evaluate_lambda(expression, state, tools)
    elif isinstance(expression, ast.Set):
        return {evaluate_ast(elt, state, tools) for elt in expression.elts}
    elif isinstance(expression, ast.SetComp):
        return evaluate_setcomp(expression, state, tools)
    elif isinstance(expression, ast.DictComp):
        return evaluate_dictcomp(expression, state, tools)
    elif isinstance(expression, ast.GeneratorExp):
        return evaluate_generator_exp(expression, state, tools)
    elif isinstance(expression, ast.Yield):
        return evaluate_yield(expression, state, tools)
    elif isinstance(expression, ast.YieldFrom):
        return evaluate_yield_from(expression, state, tools)
    elif isinstance(expression, ast.With):
        return evaluate_with(expression, state, tools)
    elif isinstance(expression, ast.Delete):
        return evaluate_delete(expression, state, tools)
    elif isinstance(expression, ast.Global):
        return evaluate_global(expression, state, tools)
    elif isinstance(expression, ast.Nonlocal):
        return evaluate_nonlocal(expression, state, tools)
    elif isinstance(expression, ast.AsyncFunctionDef):
        return evaluate_async_function_def(expression, state, tools)
    elif isinstance(expression, ast.Await):
        return evaluate_await(expression, state, tools)
    elif isinstance(expression, ast.AsyncFor):
        return evaluate_async_for(expression, state, tools)
    elif isinstance(expression, ast.AsyncWith):
        return evaluate_async_with(expression, state, tools)
    
    # Handle index for older Python versions
    elif hasattr(ast, "Index") and isinstance(expression, ast.Index):
        return evaluate_ast(expression.value, state, tools)
    
    else:
        raise InterpretorError(f"{expression.__class__.__name__} is not supported yet.")

# Exception classes for control flow
class BreakException(Exception):
    pass

class ContinueException(Exception):
    pass

class ReturnException(Exception):
    def __init__(self, value=None):
        self.value = value


def evaluate_while(while_stmt: ast.While, state: Dict[str, Any], tools: Dict[str, Callable]):
    """Evaluate while loops with proper break/continue handling"""
    result = None
    broke_out = False
    
    try:
        while evaluate_condition(while_stmt.test, state, tools):
            try:
                for stmt in while_stmt.body:
                    line_result = evaluate_ast(stmt, state, tools)
                    if line_result is not None:
                        result = line_result
            except BreakException:
                broke_out = True
                break
            except ContinueException:
                continue
    except BreakException:
        broke_out = True
    
    # Execute else clause if loop completed normally (no break)
    if while_stmt.orelse and not broke_out:
        for stmt in while_stmt.orelse:
            line_result = evaluate_ast(stmt, state, tools)
            if line_result is not None:
                result = line_result
    
    return result

def evaluate_class_def(class_def: ast.ClassDef, state: Dict[str, Any], tools: Dict[str, Callable]):
    """Enhanced class definition evaluation with proper inheritance and decorators"""
    class_name = class_def.name
    
    # Evaluate base classes
    bases = []
    for base in class_def.bases:
        base_class = evaluate_ast(base, state, tools)
        bases.append(base_class)
    
    # Evaluate keyword arguments (metaclass, etc.)
    class_kwargs = {}
    for keyword in class_def.keywords:
        class_kwargs[keyword.arg] = evaluate_ast(keyword.value, state, tools)
    
    # Create class namespace with access to current state
    class_namespace = {'__module__': '__main__'}
    class_namespace.update(state)  # Allow access to outer scope
    
    # Execute class body in the class namespace
    for stmt in class_def.body:
        if isinstance(stmt, ast.FunctionDef):
            # Handle method definitions
            evaluate_function_def(stmt, class_namespace, tools)
        else:
            result = evaluate_ast(stmt, class_namespace, tools)
            if result is not None and isinstance(stmt, ast.Assign):
                # Class variables
                pass
    
    # Remove inherited items that shouldn't be in class dict
    class_dict = {k: v for k, v in class_namespace.items() 
                  if k not in state or k.startswith('__')}
    
    # Create the class
    metaclass = class_kwargs.pop('metaclass', type)
    new_class = metaclass(class_name, tuple(bases), class_dict, **class_kwargs)
    
    # Apply decorators if any
    for decorator in reversed(class_def.decorator_list):
        decorator_func = evaluate_ast(decorator, state, tools)
        new_class = decorator_func(new_class)
    
    # Add to state
    state[class_name] = new_class
    tools[class_name] = new_class
    
    return new_class

def evaluate_lambda(lambda_expr: ast.Lambda, state: Dict[str, Any], tools: Dict[str, Callable]):
    """Enhanced lambda evaluation with proper argument handling"""
    # Get all argument information
    args = lambda_expr.args
    param_names = [arg.arg for arg in args.args]
    defaults = [evaluate_ast(default, state, tools) for default in args.defaults]
    
    # Handle *args and **kwargs
    vararg = args.vararg.arg if args.vararg else None
    kwarg = args.kwarg.arg if args.kwarg else None
    
    # Handle keyword-only arguments
    kwonlyargs = [arg.arg for arg in args.kwonlyargs]
    kw_defaults = {}
    for i, default in enumerate(args.kw_defaults):
        if default is not None:
            kw_defaults[kwonlyargs[i]] = evaluate_ast(default, state, tools)
    
    def lambda_func(*args_vals, **kwargs):
        local_state = state.copy()
        
        # Handle positional arguments
        for i, (param, arg_val) in enumerate(zip(param_names, args_vals)):
            local_state[param] = arg_val
        
        # Handle excess positional arguments (*args)
        if vararg and len(args_vals) > len(param_names):
            local_state[vararg] = args_vals[len(param_names):]
        elif len(args_vals) > len(param_names) and not vararg:
            raise InterpretorError(f"Too many positional arguments for lambda")
        
        # Handle keyword arguments
        remaining_kwargs = kwargs.copy()
        for param in param_names:
            if param in remaining_kwargs:
                local_state[param] = remaining_kwargs.pop(param)
        
        # Handle keyword-only arguments
        for param in kwonlyargs:
            if param in remaining_kwargs:
                local_state[param] = remaining_kwargs.pop(param)
            elif param in kw_defaults:
                local_state[param] = kw_defaults[param]
        
        # Handle **kwargs
        if kwarg:
            local_state[kwarg] = remaining_kwargs
        elif remaining_kwargs:
            raise InterpretorError(f"Unexpected keyword arguments: {list(remaining_kwargs.keys())}")
        
        # Handle defaults for positional arguments
        num_required = len(param_names) - len(defaults)
        provided_args = min(len(args_vals), len(param_names))
        
        for i, default in enumerate(defaults):
            param_idx = num_required + i
            if param_idx >= provided_args:
                param = param_names[param_idx]
                if param not in local_state:
                    local_state[param] = default
        
        return evaluate_ast(lambda_expr.body, local_state, tools)
    
    return lambda_func

def evaluate_with(with_stmt: ast.With, state: Dict[str, Any], tools: Dict[str, Callable]):
    """Enhanced with statement evaluation with proper context manager support"""
    result = None
    context_managers = []
    
    try:
        # Enter all context managers
        for item in with_stmt.items:
            context_expr = evaluate_ast(item.context_expr, state, tools)
            
            # Check if it's a proper context manager
            if hasattr(context_expr, '__enter__') and hasattr(context_expr, '__exit__'):
                enter_result = context_expr.__enter__()
                context_managers.append((context_expr, enter_result))
                
                # Assign to optional variable
                if item.optional_vars:
                    if isinstance(item.optional_vars, ast.Name):
                        state[item.optional_vars.id] = enter_result
                    elif isinstance(item.optional_vars, ast.Tuple):
                        # Handle tuple unpacking
                        if isinstance(enter_result, (list, tuple)):
                            for target, value in zip(item.optional_vars.elts, enter_result):
                                state[target.id] = value
            else:
                # For non-context managers, just assign the value
                context_managers.append((None, context_expr))
                if item.optional_vars:
                    state[item.optional_vars.id] = context_expr
        
        # Execute body
        for stmt in with_stmt.body:
            line_result = evaluate_ast(stmt, state, tools)
            if line_result is not None:
                result = line_result
                
    except Exception as e:
        # Handle exceptions in context managers
        for context_manager, _ in reversed(context_managers):
            if context_manager and hasattr(context_manager, '__exit__'):
                exc_type, exc_value, exc_tb = sys.exc_info()
                if not context_manager.__exit__(exc_type, exc_value, exc_tb):
                    raise
        raise
    else:
        # Normal exit
        for context_manager, _ in reversed(context_managers):
            if context_manager and hasattr(context_manager, '__exit__'):
                context_manager.__exit__(None, None, None)
    
    return result

def evaluate_global(global_stmt: ast.Global, state: Dict[str, Any], tools: Dict[str, Callable]):
    """Enhanced global statement evaluation"""
    # In a simplified interpreter, we mark variables as global
    # This would need proper scope handling in a full implementation
    for name in global_stmt.names:
        # Mark variable as global by adding a special marker
        # In practice, this would affect variable lookup in nested scopes
        state[f'__global__{name}'] = True
    return None

def evaluate_nonlocal(nonlocal_stmt: ast.Nonlocal, state: Dict[str, Any], tools: Dict[str, Callable]):
    """Enhanced nonlocal statement evaluation"""
    # Similar to global, but for nonlocal scope
    for name in nonlocal_stmt.names:
        state[f'__nonlocal__{name}'] = True
    return None

def evaluate_yield(yield_expr: ast.Yield, state: Dict[str, Any], tools: Dict[str, Callable]):
    """Enhanced yield evaluation for generator functions"""
    # This is a simplified implementation
    # In a full implementation, this would need proper generator support
    if yield_expr.value:
        return evaluate_ast(yield_expr.value, state, tools)
    return None

def evaluate_yield_from(yield_from: ast.YieldFrom, state: Dict[str, Any], tools: Dict[str, Callable]):
    """Enhanced yield from evaluation"""
    # Simplified implementation
    iterable = evaluate_ast(yield_from.value, state, tools)
    # In a full implementation, this would delegate to the sub-generator
    return iterable

def evaluate_function_def(function_def, state, tools):
    """Enhanced function definition with proper argument handling and decorators"""
    func_name = function_def.name
    args = function_def.args
    
    # Get all argument information
    param_names = [arg.arg for arg in args.args]
    defaults = [evaluate_ast(default, state, tools) for default in args.defaults]
    
    # Handle *args and **kwargs
    vararg = args.vararg.arg if args.vararg else None
    kwarg = args.kwarg.arg if args.kwarg else None
    
    # Handle keyword-only arguments
    kwonlyargs = [arg.arg for arg in args.kwonlyargs]
    kw_defaults = {}
    for i, default in enumerate(args.kw_defaults):
        if default is not None:
            kw_defaults[kwonlyargs[i]] = evaluate_ast(default, state, tools)
    
    body = function_def.body
    
    def func(*args_vals, **kwargs):
        local_state = state.copy()
        
        # Track which parameters have been satisfied
        satisfied_params = set()
        
        # Handle positional arguments
        for i, (param, arg_val) in enumerate(zip(param_names, args_vals)):
            local_state[param] = arg_val
            satisfied_params.add(param)
        
        # Handle excess positional arguments (*args)
        if vararg and len(args_vals) > len(param_names):
            local_state[vararg] = args_vals[len(param_names):]
        elif len(args_vals) > len(param_names) and not vararg:
            raise InterpretorError(f"Too many positional arguments for function {func_name}")
        
        # Handle keyword arguments
        remaining_kwargs = kwargs.copy()
        for param in param_names:
            if param in remaining_kwargs:
                local_state[param] = remaining_kwargs.pop(param)
                satisfied_params.add(param)
        
        # Handle keyword-only arguments
        for param in kwonlyargs:
            if param in remaining_kwargs:
                local_state[param] = remaining_kwargs.pop(param)
                satisfied_params.add(param)
            elif param in kw_defaults:
                local_state[param] = kw_defaults[param]
                satisfied_params.add(param)
            else:
                raise InterpretorError(f"Missing required keyword argument: {param}")
        
        # Handle **kwargs
        if kwarg:
            local_state[kwarg] = remaining_kwargs
        elif remaining_kwargs:
            raise InterpretorError(f"Unexpected keyword arguments: {list(remaining_kwargs.keys())}")
        
        # Check if all required parameters are satisfied
        num_defaults = len(defaults)
        num_required = len(param_names) - num_defaults
        
        # Apply defaults for missing positional parameters
        for i in range(num_required, len(param_names)):
            param = param_names[i]
            if param not in satisfied_params:
                default_idx = i - num_required
                if default_idx < len(defaults):
                    local_state[param] = defaults[default_idx]
                    satisfied_params.add(param)
        
        # Final check: ensure all required parameters are satisfied
        for i in range(num_required):
            param = param_names[i]
            if param not in satisfied_params:
                raise InterpretorError(f"Missing required argument: '{param}' for function {func_name}")
        
        # Execute function body
        result = None
        try:
            for node in body:
                line_result = evaluate_ast(node, local_state, tools)
                if line_result is not None:
                    result = line_result
        except ReturnException as ret:
            result = ret.value
        
        return result
    
    # Apply decorators if any
    for decorator in reversed(function_def.decorator_list):
        decorator_func = evaluate_ast(decorator, state, tools)
        func = decorator_func(func)
    
    # Add the function to state and tools
    state[func_name] = func
    tools[func_name] = func
    
    return None

def evaluate_async_function_def(async_def: ast.AsyncFunctionDef, state: Dict[str, Any], tools: Dict[str, Callable]):
    """Enhanced async function definition (simplified for sync execution)"""
    # For simplicity, treat as regular function but mark as async
    result = evaluate_function_def(async_def, state, tools)
    
    # Wrap the function to indicate it's async
    original_func = state[async_def.name]
    
    def async_wrapper(*args, **kwargs):
        # In a full implementation, this would return a coroutine
        return original_func(*args, **kwargs)
    
    async_wrapper.__name__ = async_def.name
    async_wrapper._is_async = True
    
    state[async_def.name] = async_wrapper
    tools[async_def.name] = async_wrapper
    
    return None

def evaluate_await(await_expr: ast.Await, state: Dict[str, Any], tools: Dict[str, Callable]):
    """Enhanced await evaluation (simplified for sync execution)"""
    # In a full async implementation, this would await the coroutine
    value = evaluate_ast(await_expr.value, state, tools)
    
    # For simplicity, just return the value
    # In a real implementation, this would involve event loop integration
    return value

def evaluate_async_for(async_for: ast.AsyncFor, state: Dict[str, Any], tools: Dict[str, Callable]):
    """Enhanced async for loop evaluation (simplified for sync execution)"""
    # For simplicity, treat as regular for loop
    return evaluate_for(async_for, state, tools)

def evaluate_async_with(async_with: ast.AsyncWith, state: Dict[str, Any], tools: Dict[str, Callable]):
    """Enhanced async with statement evaluation (simplified for sync execution)"""
    # For simplicity, treat as regular with statement
    return evaluate_with(async_with, state, tools)


def evaluate_setcomp(setcomp: ast.SetComp, state: Dict[str, Any], tools: Dict[str, Callable]):
    """Evaluate set comprehensions"""
    def process_generators(generators, index=0):
        if index >= len(generators):
            yield evaluate_ast(setcomp.elt, state, tools)
            return
        
        generator = generators[index]
        iter_obj = evaluate_ast(generator.iter, state, tools)
        
        for item in iter_obj:
            old_values = {}
            
            # Set loop variable
            if isinstance(generator.target, ast.Tuple):
                for target, val in zip(generator.target.elts, item):
                    old_values[target.id] = state.get(target.id)
                    state[target.id] = val
            else:
                old_values[generator.target.id] = state.get(generator.target.id)
                state[generator.target.id] = item
            
            # Check conditions
            if all(evaluate_ast(if_clause, state, tools) for if_clause in generator.ifs):
                yield from process_generators(generators, index + 1)
            
            # Restore old values
            for var, old_val in old_values.items():
                if old_val is None and var in state:
                    del state[var]
                elif old_val is not None:
                    state[var] = old_val
    
    return set(process_generators(setcomp.generators))

def evaluate_dictcomp(dictcomp: ast.DictComp, state: Dict[str, Any], tools: Dict[str, Callable]):
    """Evaluate dictionary comprehensions"""
    def process_generators(generators, index=0):
        if index >= len(generators):
            key = evaluate_ast(dictcomp.key, state, tools)
            value = evaluate_ast(dictcomp.value, state, tools)
            yield (key, value)
            return
        
        generator = generators[index]
        iter_obj = evaluate_ast(generator.iter, state, tools)
        
        for item in iter_obj:
            old_values = {}
            
            # Set loop variable
            if isinstance(generator.target, ast.Tuple):
                for target, val in zip(generator.target.elts, item):
                    old_values[target.id] = state.get(target.id)
                    state[target.id] = val
            else:
                old_values[generator.target.id] = state.get(generator.target.id)
                state[generator.target.id] = item
            
            # Check conditions
            if all(evaluate_ast(if_clause, state, tools) for if_clause in generator.ifs):
                yield from process_generators(generators, index + 1)
            
            # Restore old values
            for var, old_val in old_values.items():
                if old_val is None and var in state:
                    del state[var]
                elif old_val is not None:
                    state[var] = old_val
    
    return dict(process_generators(dictcomp.generators))

def evaluate_generator_exp(genexp: ast.GeneratorExp, state: Dict[str, Any], tools: Dict[str, Callable]):
    """Evaluate generator expressions"""
    def generator():
        def process_generators(generators, index=0):
            if index >= len(generators):
                yield evaluate_ast(genexp.elt, state, tools)
                return
            
            generator_def = generators[index]
            iter_obj = evaluate_ast(generator_def.iter, state, tools)
            
            for item in iter_obj:
                old_values = {}
                
                # Set loop variable
                if isinstance(generator_def.target, ast.Tuple):
                    for target, val in zip(generator_def.target.elts, item):
                        old_values[target.id] = state.get(target.id)
                        state[target.id] = val
                else:
                    old_values[generator_def.target.id] = state.get(generator_def.target.id)
                    state[generator_def.target.id] = item
                
                # Check conditions
                if all(evaluate_ast(if_clause, state, tools) for if_clause in generator_def.ifs):
                    yield from process_generators(generators, index + 1)
                
                # Restore old values
                for var, old_val in old_values.items():
                    if old_val is None and var in state:
                        del state[var]
                    elif old_val is not None:
                        state[var] = old_val
        
        yield from process_generators(genexp.generators)
    
    return generator()

def evaluate_delete(delete_stmt: ast.Delete, state: Dict[str, Any], tools: Dict[str, Callable]):
    """Enhanced delete statement evaluation"""
    for target in delete_stmt.targets:
        if isinstance(target, ast.Name):
            var_name = target.id
            if var_name in state:
                del state[var_name]
            if var_name in tools:
                del tools[var_name]
        elif isinstance(target, ast.Subscript):
            obj = evaluate_ast(target.value, state, tools)
            key = evaluate_ast(target.slice, state, tools)
            del obj[key]
        elif isinstance(target, ast.Attribute):
            obj = evaluate_ast(target.value, state, tools)
            delattr(obj, target.attr)
        elif isinstance(target, ast.Tuple):
            # Handle tuple deletion
            for elt in target.elts:
                evaluate_delete(ast.Delete(targets=[elt]), state, tools)
    
    return None

def evaluate_listcomp(listcomp: ast.ListComp, state: Dict[str, Any], tools: Dict[str, Callable]):
    """Enhanced list comprehension with multiple generators and conditions"""
    def process_generators(generators, index=0):
        if index >= len(generators):
            yield evaluate_ast(listcomp.elt, state, tools)
            return
        
        generator = generators[index]
        iter_obj = evaluate_ast(generator.iter, state, tools)
        
        for item in iter_obj:
            old_values = {}
            
            # Set loop variable
            if isinstance(generator.target, ast.Tuple):
                for target, val in zip(generator.target.elts, item):
                    old_values[target.id] = state.get(target.id)
                    state[target.id] = val
            else:
                old_values[generator.target.id] = state.get(generator.target.id)
                state[generator.target.id] = item
            
            # Check conditions
            if all(evaluate_ast(if_clause, state, tools) for if_clause in generator.ifs):
                yield from process_generators(generators, index + 1)
            
            # Restore old values
            for var, old_val in old_values.items():
                if old_val is None and var in state:
                    del state[var]
                elif old_val is not None:
                    state[var] = old_val
    
    return list(process_generators(listcomp.generators))

def evaluate_try(try_stmt: ast.Try, state: Dict[str, Any], tools: Dict[str, Callable]):
    """Enhanced try/except with proper exception handling"""
    result = None
    exception_caught = False
    raised_exception = None
    
    try:
        for stmt in try_stmt.body:
            line_result = evaluate_ast(stmt, state, tools)
            if line_result is not None:
                result = line_result
    except Exception as e:
        exception_caught = True
        raised_exception = e
        handled = False
        
        for handler in try_stmt.handlers:
            if handler.type is None:  # bare except
                handled = True
            else:
                # Check if exception matches handler type
                try:
                    exception_type = evaluate_ast(handler.type, state, tools)
                    if isinstance(e, exception_type):
                        handled = True
                except:
                    # If we can't evaluate the exception type, skip this handler
                    continue
            
            if handled:
                if handler.name:
                    state[handler.name] = e
                
                for stmt in handler.body:
                    line_result = evaluate_ast(stmt, state, tools)
                    if line_result is not None:
                        result = line_result
                break
        
        if not handled:
            raise
    else:
        # Execute else block if no exception
        if try_stmt.orelse:
            for stmt in try_stmt.orelse:
                line_result = evaluate_ast(stmt, state, tools)
                if line_result is not None:
                    result = line_result
    finally:
        # Always execute finally block
        if try_stmt.finalbody:
            for stmt in try_stmt.finalbody:
                evaluate_ast(stmt, state, tools)
    
    return result

def evaluate_assert(assert_statement: ast.Assert, state: Dict[str, Any], tools: Dict[str, Callable]):
    condition = evaluate_ast(assert_statement.test, state, tools)
    if not condition:
        if assert_statement.msg is not None:
            msg = evaluate_ast(assert_statement.msg, state, tools)
            raise AssertionError(msg)
        else:
            raise AssertionError("Assertion failed")
    return None

def evaluate_raise(raise_expr: ast.Raise, state: Dict[str, Any], tools: Dict[str, Callable]):
    if raise_expr.exc is not None:
        exception = evaluate_ast(raise_expr.exc, state, tools)
        if isinstance(exception, BaseException):
            raise exception
        elif isinstance(exception, type) and issubclass(exception, BaseException):
            raise exception()
        else:
            raise InterpretorError(f"Invalid exception: {exception}")
    else:
        # Re-raise last exception (simplified)
        raise InterpretorError("Re-raise without active exception is not fully supported")

def evaluate_return(return_statement: ast.Return, state: Dict[str, Any], tools: Dict[str, Callable]):
    if return_statement.value is not None:
        value = evaluate_ast(return_statement.value, state, tools)
        raise ReturnException(value)
    else:
        raise ReturnException(None)

def evaluate_boolop(boolop, state, tools):
    if isinstance(boolop.op, ast.And):
        for value in boolop.values:
            result = evaluate_ast(value, state, tools)
            if not result:
                return result
        return result
    elif isinstance(boolop.op, ast.Or):
        for value in boolop.values:
            result = evaluate_ast(value, state, tools)
            if result:
                return result
        return result
    else:
        raise InterpretorError(f"Boolean operator {boolop.op} not supported")

def evaluate_compare(compare, state, tools):
    left = evaluate_ast(compare.left, state, tools)
    
    for op, comparator in zip(compare.ops, compare.comparators):
        right = evaluate_ast(comparator, state, tools)
        
        if isinstance(op, ast.Eq):
            result = left == right
        elif isinstance(op, ast.NotEq):
            result = left != right
        elif isinstance(op, ast.Lt):
            result = left < right
        elif isinstance(op, ast.LtE):
            result = left <= right
        elif isinstance(op, ast.Gt):
            result = left > right
        elif isinstance(op, ast.GtE):
            result = left >= right
        elif isinstance(op, ast.In):
            result = left in right
        elif isinstance(op, ast.NotIn):
            result = left not in right
        elif isinstance(op, ast.Is):
            result = left is right
        elif isinstance(op, ast.IsNot):
            result = left is not right
        else:
            raise InterpretorError(f"Comparison operator {op} not supported")
        
        if not result:
            return False
        
        left = right  # For chained comparisons
    
    return True

def evaluate_import(import_node, state):
    for alias in import_node.names:
        try:
            module = __import__(alias.name)
            name = alias.asname if alias.asname else alias.name
            state[name] = module
        except ImportError as e:
            raise InterpretorError(f"Cannot import module {alias.name}: {e}")
    return None

def evaluate_import_from(import_from_node, state):
    try:
        module = __import__(import_from_node.module, fromlist=[alias.name for alias in import_from_node.names])
        for alias in import_from_node.names:
            if alias.name == '*':
                # Handle from module import *
                if hasattr(module, '__all__'):
                    names = module.__all__
                else:
                    names = [name for name in dir(module) if not name.startswith('_')]
                for name in names:
                    state[name] = getattr(module, name)
            else:
                name = alias.asname if alias.asname else alias.name
                state[name] = getattr(module, alias.name)
    except (ImportError, AttributeError) as e:
        raise InterpretorError(f"Cannot import from module {import_from_node.module}: {e}")
    return None

def evaluate_unaryop(unaryop, state, tools):
    operand = evaluate_ast(unaryop.operand, state, tools)
    if isinstance(unaryop.op, ast.UAdd):
        return +operand
    elif isinstance(unaryop.op, ast.USub):
        return -operand
    elif isinstance(unaryop.op, ast.Not):
        return not operand
    elif isinstance(unaryop.op, ast.Invert):
        return ~operand
    else:
        raise InterpretorError(f"Unary operator {unaryop.op} not supported")

def evaluate_aug_assign(aug_assign, state, tools):
    if isinstance(aug_assign.target, ast.Name):
        target_name = aug_assign.target.id
        current_value = state.get(target_name)
        if current_value is None:
            raise InterpretorError(f"Variable {target_name} not defined for augmented assignment")
    elif isinstance(aug_assign.target, ast.Subscript):
        obj = evaluate_ast(aug_assign.target.value, state, tools)
        key = evaluate_ast(aug_assign.target.slice, state, tools)
        current_value = obj[key]
    elif isinstance(aug_assign.target, ast.Attribute):
        obj = evaluate_ast(aug_assign.target.value, state, tools)
        current_value = getattr(obj, aug_assign.target.attr)
    else:
        raise InterpretorError(f"Unsupported augmented assignment target: {type(aug_assign.target)}")
    
    rhs_value = evaluate_ast(aug_assign.value, state, tools)
    
    if isinstance(aug_assign.op, ast.Add):
        result = current_value + rhs_value
    elif isinstance(aug_assign.op, ast.Sub):
        result = current_value - rhs_value
    elif isinstance(aug_assign.op, ast.Mult):
        result = current_value * rhs_value
    elif isinstance(aug_assign.op, ast.Div):
        result = current_value / rhs_value
    elif isinstance(aug_assign.op, ast.FloorDiv):
        result = current_value // rhs_value
    elif isinstance(aug_assign.op, ast.Mod):
        result = current_value % rhs_value
    elif isinstance(aug_assign.op, ast.Pow):
        result = current_value ** rhs_value
    elif isinstance(aug_assign.op, ast.LShift):
        result = current_value << rhs_value
    elif isinstance(aug_assign.op, ast.RShift):
        result = current_value >> rhs_value
    elif isinstance(aug_assign.op, ast.BitOr):
        result = current_value | rhs_value
    elif isinstance(aug_assign.op, ast.BitXor):
        result = current_value ^ rhs_value
    elif isinstance(aug_assign.op, ast.BitAnd):
        result = current_value & rhs_value
    else:
        raise InterpretorError(f"Augmented assignment operator {aug_assign.op} not supported")
    
    # Assign the result back
    if isinstance(aug_assign.target, ast.Name):
        state[aug_assign.target.id] = result
    elif isinstance(aug_assign.target, ast.Subscript):
        obj = evaluate_ast(aug_assign.target.value, state, tools)
        key = evaluate_ast(aug_assign.target.slice, state, tools)
        obj[key] = result
    elif isinstance(aug_assign.target, ast.Attribute):
        obj = evaluate_ast(aug_assign.target.value, state, tools)
        setattr(obj, aug_assign.target.attr, result)
    
    return result

def evaluate_ifexp(ifexp, state, tools):
    condition = evaluate_ast(ifexp.test, state, tools)
    if condition:
        return evaluate_ast(ifexp.body, state, tools)
    else:
        return evaluate_ast(ifexp.orelse, state, tools)

def evaluate_assign(assign, state, tools):
    result = evaluate_ast(assign.value, state, tools)
    
    for target in assign.targets:
        if isinstance(target, ast.Tuple):
            # Tuple unpacking
            if not isinstance(result, (list, tuple)):
                raise InterpretorError(f"Cannot unpack non-sequence {type(result)}")
            if len(result) != len(target.elts):
                raise InterpretorError(f"Expected {len(target.elts)} values but got {len(result)}")
            for var, value in zip(target.elts, result):
                if isinstance(var, ast.Name):
                    state[var.id] = value
                else:
                    raise InterpretorError(f"Unsupported unpacking target: {type(var)}")
        elif isinstance(target, ast.Name):
            state[target.id] = result
        elif isinstance(target, ast.Subscript):
            obj = evaluate_ast(target.value, state, tools)
            key = evaluate_ast(target.slice, state, tools)
            obj[key] = result
        elif isinstance(target, ast.Attribute):
            obj = evaluate_ast(target.value, state, tools)
            setattr(obj, target.attr, result)
        else:
            raise InterpretorError(f"Unsupported assignment target: {type(target)}")
    
    return result

def evaluate_attribute(attribute: ast.Attribute, state: Dict[str, Any], tools: Dict[str, Callable]):
    obj = evaluate_ast(attribute.value, state, tools)
    attr_name = attribute.attr
    
    try:
        return getattr(obj, attr_name)
    except AttributeError:
        raise InterpretorError(f"'{type(obj).__name__}' object has no attribute '{attr_name}'")

def evaluate_call(call, state, tools):
    if isinstance(call.func, ast.Attribute):
        obj = evaluate_ast(call.func.value, state, tools)
        method_name = call.func.attr
        try:
            method = getattr(obj, method_name)
        except AttributeError:
            raise InterpretorError(f"'{type(obj).__name__}' object has no attribute '{method_name}'")
    elif isinstance(call.func, ast.Name):
        func_name = call.func.id
        if func_name in tools:
            method = tools[func_name]
        elif func_name in state:
            method = state[func_name]
        else:
            raise InterpretorError(f"Function '{func_name}' is not defined")
    else:
        # Handle other callable expressions
        method = evaluate_ast(call.func, state, tools)
    
    # Handle arguments
    args = []
    for arg in call.args:
        if isinstance(arg, ast.Starred):
            args.extend(evaluate_ast(arg.value, state, tools))
        else:
            args.append(evaluate_ast(arg, state, tools))
    
    kwargs = {}
    for keyword in call.keywords:
        if keyword.arg is None:  # **kwargs
            kwargs.update(evaluate_ast(keyword.value, state, tools))
        else:
            kwargs[keyword.arg] = evaluate_ast(keyword.value, state, tools)
    
    try:
        return method(*args, **kwargs)
    except TypeError as e:
        raise InterpretorError(f"Error calling function: {e}")

def evaluate_subscript(subscript, state, tools):
    value = evaluate_ast(subscript.value, state, tools)
    
    if isinstance(subscript.slice, ast.Slice):
        sliced = evaluate_slice(subscript.slice, state, tools)
    else:
        sliced = evaluate_ast(subscript.slice, state, tools)
    
    try:
        return value[sliced]
    except (KeyError, IndexError, TypeError):
        if isinstance(sliced, str) and isinstance(value, Mapping):
            close_matches = difflib.get_close_matches(sliced, list(value.keys()))
            if close_matches:
                return value[close_matches[0]]
        raise InterpretorError(f"Cannot index {type(value).__name__} with {type(sliced).__name__}")

def evaluate_name(name, state, tools):
    var_name = name.id
    
    if var_name in state:
        return state[var_name]
    elif var_name in tools:
        return tools[var_name]
    else:
        # Try to find close matches
        all_names = list(state.keys()) + list(tools.keys())
        close_matches = difflib.get_close_matches(var_name, all_names)
        if close_matches:
            return state.get(close_matches[0]) or tools.get(close_matches[0])
        raise InterpretorError(f"Variable '{var_name}' is not defined")

def evaluate_condition(condition, state, tools):
    if hasattr(condition, "ops") and condition.ops:
        if len(condition.ops) > 1:
            # Handle chained comparisons
            return evaluate_compare(condition, state, tools)
        
        left = evaluate_ast(condition.left, state, tools)
        comparator = condition.ops[0]
        right = evaluate_ast(condition.comparators[0], state, tools)
        
        if isinstance(comparator, ast.Eq):
            return left == right
        elif isinstance(comparator, ast.NotEq):
            return left != right
        elif isinstance(comparator, ast.Lt):
            return left < right
        elif isinstance(comparator, ast.LtE):
            return left <= right
        elif isinstance(comparator, ast.Gt):
            return left > right
        elif isinstance(comparator, ast.GtE):
            return left >= right
        elif isinstance(comparator, ast.Is):
            return left is right
        elif isinstance(comparator, ast.IsNot):
            return left is not right
        elif isinstance(comparator, ast.In):
            return left in right
        elif isinstance(comparator, ast.NotIn):
            return left not in right
        else:
            raise InterpretorError(f"Comparison operator {comparator} not supported")
    else:
        return bool(evaluate_ast(condition, state, tools))

def evaluate_if(if_statement, state, tools):
    result = None
    if evaluate_condition(if_statement.test, state, tools):
        for stmt in if_statement.body:
            line_result = evaluate_ast(stmt, state, tools)
            if line_result is not None:
                result = line_result
    else:
        for stmt in if_statement.orelse:
            line_result = evaluate_ast(stmt, state, tools)
            if line_result is not None:
                result = line_result
    return result

def unpack_tuple(target, values, state):
    if len(values) != len(target.elts):
        raise InterpretorError("Mismatch in number of variables to unpack")
    
    for idx, variable in enumerate(target.elts):
        if isinstance(variable, ast.Tuple):
            unpack_tuple(variable, values[idx], state)
        else:
            state[variable.id] = values[idx]

def evaluate_for(for_loop, state, tools):
    result = None
    iterator = evaluate_ast(for_loop.iter, state, tools)
    broke_out = False
    
    try:
        if isinstance(for_loop.target, ast.Tuple):
            for values in iterator:
                try:
                    unpack_tuple(for_loop.target, values, state)
                    
                    for expression in for_loop.body:
                        line_result = evaluate_ast(expression, state, tools)
                        if line_result is not None:
                            result = line_result
                except BreakException:
                    broke_out = True
                    break
                except ContinueException:
                    continue
        else:
            for counter in iterator:
                try:
                    state[for_loop.target.id] = counter
                    
                    for expression in for_loop.body:
                        line_result = evaluate_ast(expression, state, tools)
                        if line_result is not None:
                            result = line_result
                except BreakException:
                    broke_out = True
                    break
                except ContinueException:
                    continue
    except BreakException:
        broke_out = True
    
    # Execute else clause if loop completed normally (no break)
    if for_loop.orelse and not broke_out:
        for stmt in for_loop.orelse:
            line_result = evaluate_ast(stmt, state, tools)
            if line_result is not None:
                result = line_result
    
    return result

def evaluate_binop(binop, state, tools):
    left = evaluate_ast(binop.left, state, tools)
    right = evaluate_ast(binop.right, state, tools)
    
    if isinstance(binop.op, ast.Add):
        return left + right
    elif isinstance(binop.op, ast.Sub):
        return left - right
    elif isinstance(binop.op, ast.Mult):
        return left * right
    elif isinstance(binop.op, ast.Div):
        return left / right
    elif isinstance(binop.op, ast.FloorDiv):
        return left // right
    elif isinstance(binop.op, ast.Mod):
        return left % right
    elif isinstance(binop.op, ast.Pow):
        return left ** right
    elif isinstance(binop.op, ast.LShift):
        return left << right
    elif isinstance(binop.op, ast.RShift):
        return left >> right
    elif isinstance(binop.op, ast.BitOr):
        return left | right
    elif isinstance(binop.op, ast.BitXor):
        return left ^ right
    elif isinstance(binop.op, ast.BitAnd):
        return left & right
    else:
        raise InterpretorError(f"Binary operator {binop.op} not supported")

def evaluate_slice(slice_op, state, tools):
    lower = evaluate_ast(slice_op.lower, state, tools) if slice_op.lower else None
    upper = evaluate_ast(slice_op.upper, state, tools) if slice_op.upper else None
    step = evaluate_ast(slice_op.step, state, tools) if slice_op.step else None
    return slice(lower, upper, step)

def create_persistent_interpreter():
    """Create a persistent interpreter instance with enhanced built-ins"""
    tools = {
        'print': print,
        'len': len,
        'str': str,
        'int': int,
        'float': float,
        'bool': bool,
        'list': list,
        'dict': dict,
        'set': set,
        'tuple': tuple,
        'range': range,
        'enumerate': enumerate,
        'zip': zip,
        'map': map,
        'filter': filter,
        'sum': sum,
        'max': max,
        'min': min,
        'abs': abs,
        'round': round,
        'sorted': sorted,
        'reversed': reversed,
        'type': type,
        'isinstance': isinstance,
        'issubclass': issubclass,
        'hasattr': hasattr,
        'getattr': getattr,
        'setattr': setattr,
        'delattr': delattr,
        'dir': dir,
        'vars': vars,
        'callable': callable,
        'iter': iter,
        'next': next,
        'all': all,
        'any': any,
        'chr': chr,
        'ord': ord,
        'hex': hex,
        'oct': oct,
        'bin': bin,
        'divmod': divmod,
        'pow': pow,
        'repr': repr,
        'slice': slice,
        # Exception types
        'Exception': Exception,
        'ValueError': ValueError,
        'TypeError': TypeError,
        'KeyError': KeyError,
        'IndexError': IndexError,
        'AttributeError': AttributeError,
        'ImportError': ImportError,
        'AssertionError': AssertionError,
        'RuntimeError': RuntimeError,
        'NotImplementedError': NotImplementedError,
    }
    
    return PersistentInterpreter(tools)

if __name__ == "__main__":
    interpreter = create_persistent_interpreter()
    
    # Test examples
    print("Testing enhanced interpreter...")
    
    # Test function with decorators and complex arguments
    code1 = """
def decorator(func):
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

@decorator
def greet(name, greeting="Hello", *, formal=False):
    if formal:
        return f"{greeting}, {name}."
    return f"{greeting} {name}!"

result = greet("World", formal=True)
print(result)
"""
    
    # Test class definition
    code2 = """
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def greet(self):
        return f"Hi, I'm {self.name} and I'm {self.age} years old"
    
    @property
    def is_adult(self):
        return self.age >= 18

person = Person("Alice", 25)
print(person.greet())
"""
    
    # Test comprehensions and generators
    code3 = """
numbers = [1, 2, 3, 4, 5]
squares = [x**2 for x in numbers if x % 2 == 0]
print(f"Squares: {squares}")

# Generator expression
gen = (x**2 for x in range(5))
print(f"Generator: {list(gen)}")

# Dictionary comprehension
square_dict = {x: x**2 for x in range(5)}
print(f"Dict: {square_dict}")
"""
    
    try:
        interpreter.execute(code1)
        print("\n" + "="*50 + "\n")
        interpreter.execute(code2)
        print("\n" + "="*50 + "\n")
        interpreter.execute(code3)
        
        print(f"\nFinal interpreter state keys: {list(interpreter.global_state.keys())}")
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()