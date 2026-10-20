import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

type Cell = string | number | null | undefined;

export function DataTable({ columns, rows }: { columns: string[]; rows: Cell[][] }) {
  return (
    <div className="overflow-x-auto rounded-lg border border-slate-700/70 bg-slate-950/45">
      <Table>
        <TableHeader>
          <TableRow>
            {columns.map((c) => (
              <TableHead key={c}>{c}</TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((r, i) => (
            <TableRow key={i}>
              {r.map((v, j) => (
                <TableCell key={j} className="px-3 py-2 text-slate-300">
                  {v ?? "-"}
                </TableCell>
              ))}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
