import gdb
import os
import shutil

SNAPSHOT_DIR = "./snapshots"

if os.path.exists(SNAPSHOT_DIR):
    shutil.rmtree(SNAPSHOT_DIR)
os.makedirs(SNAPSHOT_DIR)

def get_field_value(field):
    try:
        if field.type.code == gdb.TYPE_CODE_PTR:
            if field.is_optimized_out:
                return "<optimized out>"
            if field:
                try:
                    # Dereference the pointer to get the actual value
                    dereferenced = field.dereference()
                    if dereferenced.type.code in [gdb.TYPE_CODE_STRUCT, gdb.TYPE_CODE_UNION]:
                        # Recursively dump struct or union fields
                        return dump_struct_or_union(dereferenced)
                    elif dereferenced.type.code == gdb.TYPE_CODE_ARRAY:
                        # Handle arrays
                        return "<array of {}, size {}>".format(
                            dereferenced.type.target().name, dereferenced.type.range()[1] + 1
                        )
                    else:
                        # For other types, return the value as a string
                        return str(dereferenced)
                except gdb.error:
                    return str(field)  # If dereference fails, return the address
            return "NULL"

        if field.type.code == gdb.TYPE_CODE_ARRAY:
            return "<array of {}, size {}>".format(field.type.target().name, field.type.range()[1] + 1)

        if field.type.code in [gdb.TYPE_CODE_STRUCT, gdb.TYPE_CODE_UNION]:
            # Recursively dump struct or union fields
            return dump_struct_or_union(field)

        val = str(field)
        if val == "true":
            return True
        if val == "false":
            return False

        return val
    except gdb.error:
        return "<error reading value>"

def dump_struct_or_union(field):
    try:
        fields = field.type.fields()
        result = {}
        for subfield in fields:
            if subfield.is_base_class:
                continue
            subfield_name = subfield.name
            subfield_value = get_field_value(field[subfield_name])
            result[subfield_name] = subfield_value
        return result
    except gdb.error:
        return "<error reading struct/union>"

class DumpSolverCommand(gdb.Command):
    def __init__(self):
        super(DumpSolverCommand, self).__init__("dump_solver_attrs", gdb.COMMAND_USER)

    def invoke(self, arg, from_tty):
        self.dont_repeat()

        try:
            solver_ptr = gdb.parse_and_eval("solver")

            if solver_ptr.type.code != gdb.TYPE_CODE_PTR:
                print("Error: 'solver' is not a pointer type.")
                return

            solver_struct = solver_ptr.dereference()
            fields = solver_struct.type.fields()

            output_lines = []
            output_lines.append("Dumping attributes for 'solver' of type '{}':\n".format(solver_struct.type.name))

            timestamp_val = "unknown_timestamp"

            for field in fields:
                field_name = field.name
                if field.is_base_class:
                    continue

                field_instance = solver_struct[field_name]
                field_value = get_field_value(field_instance)

                output_lines.append("  - {}: {}".format(field_name, field_value))

                if field_name == "timestamp":
                    timestamp_val = str(field_value)

            if not os.path.exists(SNAPSHOT_DIR):
                os.makedirs(SNAPSHOT_DIR)

            filename = os.path.join(SNAPSHOT_DIR, "{}_snapshot.txt".format(timestamp_val))

            with open(filename, "w") as f:
                f.write("\n".join(output_lines))
            print("Dumped solver attributes to: {}".format(filename))

            if from_tty:
                print("Succ: {}".format(filename))

        except gdb.error as e:
            print("GDB Error: {}".format(e))
        except Exception as e:
            print("Python Error: {}".format(e))

DumpSolverCommand()