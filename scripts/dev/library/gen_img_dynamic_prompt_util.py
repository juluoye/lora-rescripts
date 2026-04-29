from __future__ import annotations

import itertools
import logging
import random
import re

logger = logging.getLogger(__name__)

# regular expression for dynamic prompt:
# starts and ends with "{" and "}"
# contains at least one variant divided by "|"
# optional framgments divided by "$$" at start
# if the first fragment is "E" or "e", enumerate all variants
# if the second fragment is a number or two numbers, repeat the variants in the range
# if the third fragment is a string, use it as a separator
RE_DYNAMIC_PROMPT = re.compile(r"\{((e|E)\$\$)?(([\d\-]+)\$\$)?(([^\|\}]+?)\$\$)?(.+?((\|).+?)*?)\}")


def handle_dynamic_prompt_variants(prompt, repeat_count, seed_random, seeds=None, logger_override=None):
    founds = list(RE_DYNAMIC_PROMPT.finditer(prompt))
    if not founds:
        return [prompt], seeds

    active_logger = logger_override or logger

    if seeds is None:
        seeds = []
    while len(seeds) < repeat_count:
        seeds.append(seed_random.randint(0, 2**32 - 1))

    prompt = prompt.replace(r"\{", "｛").replace(r"\}", "｝")

    prompts = [prompt] * repeat_count
    has_dynamic = True
    while has_dynamic:
        has_dynamic = False
        new_prompts = []
        for i, prompt in enumerate(prompts):
            seed = seeds[i] if i < len(seeds) else seeds[0]

            deepest_nest_level = 0
            nest_level = 0
            for c in prompt:
                if c == "{":
                    nest_level += 1
                    deepest_nest_level = max(deepest_nest_level, nest_level)
                elif c == "}":
                    nest_level -= 1
            if deepest_nest_level == 0:
                new_prompts.append(prompt)
                continue

            positions = []
            nest_level = 0
            start_pos = -1
            for i, c in enumerate(prompt):
                if c == "{":
                    nest_level += 1
                    if nest_level == deepest_nest_level:
                        start_pos = i
                elif c == "}":
                    if nest_level == deepest_nest_level:
                        end_pos = i + 1
                        positions.append((start_pos, end_pos))
                    nest_level -= 1

            innermost_founds = []
            for start, end in positions:
                segment = prompt[start:end]
                match = RE_DYNAMIC_PROMPT.match(segment)
                if match:
                    innermost_founds.append((match, start, end))

            if not innermost_founds:
                new_prompts.append(prompt)
                continue
            has_dynamic = True

            enumerating = False
            replacers = []
            for found, start, end in innermost_founds:
                found_enumerating = found.group(2) is not None
                enumerating = enumerating or found_enumerating

                separator = ", " if found.group(6) is None else found.group(6)
                variants = found.group(7).split("|")

                count_range = found.group(4)
                if count_range is None:
                    count_range = [1, 1]
                else:
                    count_range = count_range.split("-")
                    if len(count_range) == 1:
                        count_range = [int(count_range[0]), int(count_range[0])]
                    elif len(count_range) == 2:
                        count_range = [int(count_range[0]), int(count_range[1])]
                    else:
                        active_logger.warning(f"invalid count range: {count_range}")
                        count_range = [1, 1]
                    if count_range[0] > count_range[1]:
                        count_range = [count_range[1], count_range[0]]
                    if count_range[0] < 0:
                        count_range[0] = 0
                    if count_range[1] > len(variants):
                        count_range[1] = len(variants)

                if found_enumerating:
                    def make_replacer_enum(vari, cr, sep):
                        def replacer(rnd=random):
                            values = []
                            for count in range(cr[0], cr[1] + 1):
                                for comb in itertools.combinations(vari, count):
                                    values.append(sep.join(comb))
                            return values

                        return replacer

                    replacers.append(make_replacer_enum(variants, count_range, separator))
                else:
                    def make_replacer_single(vari, cr, sep):
                        def replacer(rnd=random):
                            count = rnd.randint(cr[0], cr[1])
                            comb = rnd.sample(vari, count)
                            return [sep.join(comb)]

                        return replacer

                    replacers.append(make_replacer_single(variants, count_range, separator))

            rnd = random.Random(seed)
            if not enumerating:
                innermost_founds.reverse()
                replacers.reverse()

                current = prompt
                for (found, start, end), replacer in zip(innermost_founds, replacers):
                    current = current[:start] + replacer(rnd)[0] + current[end:]
                new_prompts.append(current)
            else:
                processing_prompts = [prompt]
                for found, replacer in zip(founds, replacers):
                    if found.group(2) is not None:
                        replaced_prompts = []
                        for current in processing_prompts:
                            replacements = replacer(rnd)
                            for replacement in replacements:
                                replaced_prompts.append(current.replace(found.group(0), replacement, 1))
                        processing_prompts = replaced_prompts

                for found, replacer in zip(founds, replacers):
                    if found.group(2) is None:
                        for i in range(len(processing_prompts)):
                            processing_prompts[i] = processing_prompts[i].replace(found.group(0), replacer(rnd)[0], 1)

                new_prompts.extend(processing_prompts)

        prompts = new_prompts

    for i in range(len(prompts)):
        prompts[i] = prompts[i].replace("｛", "{").replace("｝", "}")
    if enumerating:
        new_seeds = []
        for _ in range(len(prompts)):
            new_seeds.append(seeds[0])
        seeds = new_seeds

    return prompts, seeds


__all__ = ["RE_DYNAMIC_PROMPT", "handle_dynamic_prompt_variants"]
